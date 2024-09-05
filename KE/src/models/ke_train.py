from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from src.models.patch import monkeypatch as make_functional
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.mmkb_dataset import CaptionDataset
from src.models.one_shot_learner import OneShotLearner
from src.utils import batch_it, multiclass_log_probs
from src.models.get_models import get_model
from src.models.multimodal_training_hparams import MultimodalTrainingHparams
from collections import OrderedDict


class MLLM_KE(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="../datasets/train.json",
        )
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default="../datasets/eval.json",
        )
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_alpha", type=float, default=1e-1)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=30000)
        parser.add_argument("--warmup_updates", type=int, default=1000)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--model_name", type=str, choices=["blip2", "minigpt4", "llava", "qwen-vl", "owl-2"], default="blip2")
        parser.add_argument("--eps", type=float, default=0.1)
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="",
        )
        parser.add_argument("--margin_lp_max", type=float, default=1e-3)
        parser.add_argument("--margin_lp_min", type=float, default=1e-7)
        parser.add_argument("--max_scale", type=float, default=1)
        parser.add_argument("--p", type=float, default=2)
        parser.add_argument(
            "--divergences", type=str, choices=["lp"], default="lp"
        )
        parser.add_argument("--use_views", action="store_true")

        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        if self.hparams.model_name == 'blip2':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                '../hparams/TRAINING/KE/blip2.yaml'
            )
            self.model, self.tokenizer = get_model(self.hparams.model_name)
            vocab_dim = self.model.opt_model.model.decoder.embed_tokens.weight.shape[0]
            embedding_dim = self.model.opt_model.model.decoder.embed_tokens.weight.shape[1]
            embedding_init = self.model.opt_model.model.decoder.embed_tokens.weight.data

        elif self.hparams.model_name == 'minigpt4':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                '../hparams/TRAINING/KE/minigpt4.yaml'
            )
            self.model, self.tokenizer = get_model(self.hparams.model_name)
            vocab_dim = self.model.llama_model.model.embed_tokens.weight.shape[0]
            embedding_dim = self.model.llama_model.model.embed_tokens.weight.shape[1]
            embedding_init = self.model.llama_model.model.embed_tokens.weight.data
            
        elif self.hparams.model_name == 'llava':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                '../hparams/TRAINING/KE/llava.yaml'
            )
            self.model, self.tokenizer = get_model(self.hparams.model_name)
            vocab_dim = self.model.model.embed_tokens.weight.shape[0]
            embedding_dim = self.model.model.embed_tokens.weight.shape[1]
            embedding_init = self.model.model.embed_tokens.weight.data
        
        elif self.hparams.model_name == 'qwen-vl':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                '../hparams/TRAINING/KE/qwenvl.yaml'
            )
            self.model, self.tokenizer = get_model(self.model_hparams.name)
            vocab_dim = self.model.transformer.wte.weight.data.shape[0]
            embedding_dim = self.model.transformer.wte.weight.data.shape[1]
            embedding_init = self.model.transformer.wte.weight.data

        elif self.hparams.model_name == 'owl-2':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                '../hparams/TRAINING/KE/owl2.yaml'
            )
            self.model, self.tokenizer = get_model(self.model_hparams.name)
            vocab_dim = self.model.base_model.embed_tokens.weight.data.shape[0]
            embedding_dim = self.model.base_model.embed_tokens.weight.data.shape[1]
            embedding_init = self.model.base_model.embed_tokens.weight.data

        else:
            raise ValueError(f"Model {self.hparams.model_name} not supported")


        self.include_params_set={
            n
            for n, _ in self.model.named_parameters()
            if any(
                e in n.lower()
                for e in self.model_hparams.inner_params
            )
        }
        print(f"include_set: {self.include_params_set}")
        
        for n, p in self.model.named_parameters():
            if n in self.include_params_set:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.learner = OneShotLearner(
            self.model,
            vocab_dim=vocab_dim,
            embedding_dim=embedding_dim,
            hidden_dim=128,
            condition_dim=1024,
            include_set=self.include_params_set,
            max_scale=self.hparams.max_scale,
            embedding_init=embedding_init,
        ).to(torch.float32)

        self.alpha_lp = torch.nn.Parameter(torch.ones(()))
        self.alpha_lp.register_hook(lambda grad: -grad)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

        self.register_buffer("margin_lp", torch.tensor(self.hparams.margin_lp_max))
        self.running_flipped = []

        self.fmodel = None

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = CaptionDataset(
                data_dir=self.hparams.train_data_path, 
                config=self.model_hparams)
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self, shuffle=False):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = CaptionDataset(
                data_dir=self.hparams.dev_data_path, 
                config=self.model_hparams)
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def get_logits_orig_params_dict(self, batch):
        with torch.enable_grad():
            if self.hparams.model_name == "owl-2":
                input_ids, image = batch["edit_inner"]['input_ids'], batch["edit_inner"]['image']
                logit_for_grad= self.model.eval()(input_ids, 
                                            images=image.to(dtype=torch.float16)).logits
            else:
                logit_for_grad = self.model.eval()(
                    batch['edit_inner']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['edit_inner'],
                ).logits

            grads = torch.autograd.grad(
                    multiclass_log_probs(
                        logit_for_grad,
                        batch['edit_inner']['labels']
                    )["nll"],
                    [p for n, p in self.model.named_parameters() if n in self.include_params_set],
            )
            grad_dict = {}
            for n, grad in zip([n for n, p in self.model.named_parameters() if n in self.include_params_set], grads):
                grad_dict[n] = grad

        params_dict = self.learner(
            batch['cond']["input_ids"],
            batch['cond']["attention_mask"],
            grads=grad_dict,
        )   

        return params_dict

    def forward(self, batch):
        params_dict = self.get_logits_orig_params_dict(batch)
        self.fmodel = make_functional(self.model).eval()
        
        with torch.cuda.amp.autocast():
            if self.hparams.model_name == "owl-2":
                logits = self.fmodel(
                    batch['edit_inner']['input_ids'],
                    images = batch['edit_inner']['image'].to(dtype=torch.float16),
                    params=[
                        (params_dict.get(n, 0) + p) for n, p in self.model.named_parameters()
                    ],
                ).logits
            else:
                logits = self.fmodel(
                    batch['edit_inner']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['edit_inner'],
                    params=[
                        params_dict.get(n, 0) + p for n, p in self.model.named_parameters()
                    ],
                ).logits
        self.fmodel = None

        return logits, params_dict

    def get_kl_lp_cr(self, logits, labels, params_dict):
        lp = sum(
            (p.abs() ** self.hparams.p).mean() ** (1 / self.hparams.p)
            for p in params_dict.values()
        ) / len(params_dict)

        cr = multiclass_log_probs(logits, labels)["nll"]

        return lp, cr

    def training_step(self, batch, batch_idx):

        logits, params_dict = self.forward(batch)

        lp, cr = self.get_kl_lp_cr(
            logits, batch['edit_inner']['labels'], params_dict
        )
        
        loss_lp = self.alpha_lp * (lp - self.margin_lp)
        loss = cr + loss_lp

        self.log("alpha_lp", self.alpha_lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("lp", lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("cr", cr, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        params_dict = self.get_logits_orig_params_dict(batch)
        self.fmodel = make_functional(self.model).eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if self.hparams.model_name == "owl-2":
                    input_ids, image = batch["edit_inner"]['input_ids'], batch["edit_inner"]['image']
                    logits = self.fmodel(
                                input_ids, 
                                images=image.to(dtype=torch.float16),
                                params=[params_dict.get(n, 0) + p for n, p in self.model.named_parameters()],
                            ).logits
                else:
                    logits = self.fmodel(
                        batch['edit_inner']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['edit_inner'],
                        params=[params_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
                    ).logits

        results = multiclass_log_probs(
                logits,
                batch['edit_inner']['labels']
            )
        self.fmodel = None
        
        pred = results["pred_ids"]
        trg = results["targ_ids"]
        
        self.valid_acc(pred, trg)
        self.log(
            "valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        
    def on_before_zero_grad(self, optimizer):
        self.alpha_lp.data = torch.where(
            self.alpha_lp.data < 0,
            torch.full_like(self.alpha_lp.data, 0),
            self.alpha_lp.data,
        )

    def on_validation_epoch_end(self):
        if self.valid_acc.compute().item() > 0.9:
            self.margin_lp = max(
                self.margin_lp * 0.8, self.margin_lp * 0 + self.hparams.margin_lp_min
            )
        self.log(
            "margin_lp", self.margin_lp, on_step=False, on_epoch=True, prog_bar=True
        )
        self.fmodel = None

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            [
                {
                    "params": self.learner.parameters(),
                    "lr": self.hparams.lr,
                },
                {
                    "params": [self.alpha_lp],
                    "lr": self.hparams.lr_alpha,
                },
            ],
            centered=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # remove model weights that not trainable
        model_state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        keys_to_remove = ['fmodel.', 'model.']
        for key, value in model_state_dict.items():
            if not any(key.startswith(prefix) for prefix in keys_to_remove):
                new_state_dict[key] = value
        checkpoint['state_dict'] = new_state_dict
        return super().on_save_checkpoint(checkpoint)
