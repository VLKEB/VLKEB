from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl
from src.models.patch import monkeypatch as make_functional
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from src.data.mmkb_dataset import CaptionDataset
from src.models.one_shot_learner import OneShotLearner
from src.utils import multiclass_log_probs
from src.models.get_models import get_model
from src.models.multimodal_training_hparams import MultimodalTrainingHparams

class MLLM_KE(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default='../datasets/eval.json'
        )
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--model_name", type=str, choices=["blip2", "minigpt4", "llava", "qwen-vl", "owl-2"], default="blip2")
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="",
        )

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

        self.valid_acc =    pl.metrics.Accuracy()
        self.valid_t_gen =  pl.metrics.Accuracy()
        self.valid_m_gen =  pl.metrics.Accuracy()
        self.valid_t_loc =  pl.metrics.Accuracy()
        self.valid_m_loc =  pl.metrics.Accuracy()
        self.valid_port =  pl.metrics.Accuracy()

        self.fmodel = None

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

    def validation_step(self, batch, batch_idx=None):
        params_dict = self.get_logits_orig_params_dict(batch)
        self.fmodel = make_functional(self.model).eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                params = [params_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
                if self.hparams.model_name == "owl-2":
                    input_ids, image = batch["edit_inner"]['input_ids'], batch["edit_inner"]['image']
                    logits = self.fmodel(
                                input_ids, 
                                images=image.to(dtype=torch.float16),
                                params=params,
                            ).logits
                    results = multiclass_log_probs(logits, batch['edit_inner']['labels'])
                    self.valid_acc(results["pred_ids"], results["targ_ids"])
                    self.log("acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

                    input_ids, image = batch["edit_outer"]['input_ids'], batch["edit_outer"]['image']
                    logits = self.fmodel(
                                input_ids, 
                                images=image.to(dtype=torch.float16),
                                params=params,
                            ).logits
                    results = multiclass_log_probs(logits, batch['edit_outer']['labels'])
                    self.valid_t_gen(results["pred_ids"], results["targ_ids"])
                    self.log("t_gen", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

                    input_ids, image = batch["edit_outer_image"]['input_ids'], batch["edit_outer_image"]['image']
                    logits = self.fmodel(
                                input_ids, 
                                images=image.to(dtype=torch.float16),
                                params=params,
                            ).logits
                    results = multiclass_log_probs(logits, batch['edit_outer_image']['labels'])
                    self.valid_m_gen(results["pred_ids"], results["targ_ids"])
                    self.log("m_gen", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

                    input_ids, image = batch["loc"]['input_ids'], batch["loc"]['image']
                    base_logits = self.model.eval()(
                                input_ids, 
                                images=image.to(dtype=torch.float16),
                            ).logits
                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
                    post_base_logits = self.fmodel(
                                input_ids, 
                                images=image.to(dtype=torch.float16),
                                params=params,
                            ).logits
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                    self.valid_t_loc(base_logits_softmax_top_k.view(-1), post_base_logits_softmax_top_k.view(-1))
                    self.log("t_loc", self.valid_t_loc, on_step=False, on_epoch=True, prog_bar=True)
                            
                    input_ids, image = batch["loc_image"]['input_ids'], batch["loc_image"]['image']
                    base_image_logits = self.model.eval()(
                                input_ids, 
                                images=image.to(dtype=torch.float16),
                            ).logits
                    base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                    post_image_base_logits = self.fmodel(
                                input_ids, 
                                images=image.to(dtype=torch.float16),
                                params=params,
                            ).logits
                    post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
                    self.valid_m_loc(base_image_logits_softmax_top_k.view(-1), post_image_base_logits_softmax_top_k.view(-1))
                    self.log("m_loc", self.valid_m_loc, on_step=False, on_epoch=True, prog_bar=True)
                    
                else:
                    logits = self.fmodel(
                        batch['edit_inner']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['edit_inner'],
                        params=params,
                    ).logits
                    results = multiclass_log_probs(logits, batch['edit_inner']['labels'])
                    self.valid_acc(results["pred_ids"], results["targ_ids"])
                    self.log("acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
                    logits = None
                    results = None

                    logits = self.fmodel(
                        batch['edit_outer']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['edit_outer'],
                        params=params,
                    ).logits
                    results = multiclass_log_probs(logits, batch['edit_outer']['labels'])
                    self.valid_t_gen(results["pred_ids"], results["targ_ids"])
                    self.log("t_gen", self.valid_t_gen, on_step=False, on_epoch=True, prog_bar=True)
                    logits = None
                    results = None

                    logits = self.fmodel(
                        batch['edit_outer_image']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['edit_outer_image'], 
                        params=params
                    ).logits
                    results = multiclass_log_probs(logits, batch['edit_outer_image']['labels'])
                    self.valid_m_gen(results["pred_ids"], results["targ_ids"])
                    self.log("m_gen", self.valid_m_gen, on_step=False, on_epoch=True, prog_bar=True)
                    logits = None
                    results = None

                    base_logits = self.model.eval()(batch["loc"]['inputs'] if self.hparams.model_name == "qwen-vl" else batch['loc']).logits
                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
                    post_base_logits = self.fmodel(batch['loc']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['loc'], params=params).logits
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                    self.valid_t_loc(base_logits_softmax_top_k.view(-1), post_base_logits_softmax_top_k.view(-1))
                    self.log("t_loc", self.valid_t_loc, on_step=False, on_epoch=True, prog_bar=True)
                    base_logits = None
                    post_base_logits = None
                
                    base_image_logits = self.model.eval()(batch["loc_image"]['inputs'] if self.hparams.model_name == "qwen-vl" else batch['loc_image']).logits
                    base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                    post_image_base_logits = self.fmodel(batch['loc_image']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['loc_image'], params=params).logits
                    post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
                    self.valid_m_loc(base_image_logits_softmax_top_k.view(-1), post_image_base_logits_softmax_top_k.view(-1))
                    self.log("m_loc", self.valid_m_loc, on_step=False, on_epoch=True, prog_bar=True)
                    base_logits = None
                    post_base_logits = None

        self.fmodel = None

    def on_validation_epoch_end(self) -> None:
        self.fmodel = None
        return super().on_validation_epoch_end()