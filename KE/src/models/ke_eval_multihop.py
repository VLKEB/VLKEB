from argparse import ArgumentParser
import json
import os
import torch
import pytorch_lightning as pl
from src.models.patch import monkeypatch as make_functional
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from src.data.mmkb_dataset import MultihopCaptionDataset
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
            default='../datasets/mmkb/eval_multihop.json'
        )
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--hop", type=str, choices=['1', '2', '3', '4'])
        parser.add_argument("--model_name", type=str, choices=["blip2", "minigpt4", "llava"], default="blip2")
        parser.add_argument("--model_checkpoint", type=str, required=True)
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model, self.tokenizer = get_model(self.hparams.model_name)

        if self.hparams.model_name == 'blip2':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                '../hparams/TRAINING/KE/blip2.yaml'
            )
            vocab_dim = self.model.opt_model.model.decoder.embed_tokens.weight.shape[0]
            embedding_dim = self.model.opt_model.model.decoder.embed_tokens.weight.shape[1]
            embedding_init = self.model.opt_model.model.decoder.embed_tokens.weight.data

        elif self.hparams.model_name == 'minigpt4':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                '../hparams/TRAINING/KE/minigpt4.yaml'
            )
            vocab_dim = self.model.llama_model.model.embed_tokens.weight.shape[0]
            embedding_dim = self.model.llama_model.model.embed_tokens.weight.shape[1]
            embedding_init = self.model.llama_model.model.embed_tokens.weight.data
            
        elif self.hparams.model_name == 'llava':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                '../hparams/TRAINING/KE/llava.yaml'
            )
            vocab_dim = self.model.model.embed_tokens.weight.shape[0]
            embedding_dim = self.model.model.embed_tokens.weight.shape[1]
            embedding_init = self.model.model.embed_tokens.weight.data

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

        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.save_txt =  f'results/{cur_time}_{self.hparams.model_name}_port_hop{self.hparams.hop}.txt'
        self.save_json = f'results/{cur_time}_{self.hparams.model_name}_port_hop{self.hparams.hop}.json'
        self.port_result = []

    def val_dataloader(self, shuffle=False):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = MultihopCaptionDataset(
                data_dir=self.hparams.dev_data_path, 
                config=self.model_hparams,
                hop=self.hparams.hop)
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def get_logits_orig_params_dict(self, batch):
        with torch.enable_grad():
            logit_for_grad = self.model.eval()(
                batch['edit_inner']
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
        assert len(batch['port']) == 1, "batch['port'] should have only one element"

        params_dict = self.get_logits_orig_params_dict(batch)
        self.fmodel = make_functional(self.model).eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                params=[params_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
                
                logits = self.fmodel(batch['edit_inner'], params=params).logits
                results = multiclass_log_probs(logits, batch['edit_inner']['labels'])
                self.valid_acc(results["pred_ids"], results["targ_ids"])
                self.log("acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

                logits = self.fmodel(batch['edit_outer'], params=params).logits
                results = multiclass_log_probs(logits, batch['edit_outer']['labels'])
                self.valid_t_gen(results["pred_ids"], results["targ_ids"])
                self.log("t_gen", self.valid_t_gen, on_step=False, on_epoch=True, prog_bar=True)

                logits = self.fmodel(batch['edit_outer_image'], params=params).logits
                results = multiclass_log_probs(logits, batch['edit_outer_image']['labels'])
                self.valid_m_gen(results["pred_ids"], results["targ_ids"])
                self.log("m_gen", self.valid_m_gen, on_step=False, on_epoch=True, prog_bar=True)

                base_logits = self.model.eval()(batch["loc"]).logits
                base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
                post_base_logits = self.fmodel(batch['loc'], params=params).logits
                post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                self.valid_t_loc(base_logits_softmax_top_k.view(-1), post_base_logits_softmax_top_k.view(-1))
                self.log("t_loc", self.valid_t_loc, on_step=False, on_epoch=True, prog_bar=True)
                
                base_image_logits = self.model.eval()(batch["loc_image"]).logits
                base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                post_image_base_logits = self.fmodel(batch['loc_image'], params=params).logits
                post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
                self.valid_m_loc(base_image_logits_softmax_top_k.view(-1), post_image_base_logits_softmax_top_k.view(-1))
                self.log("m_loc", self.valid_m_loc, on_step=False, on_epoch=True, prog_bar=True)
                
                port = batch['port'][0]
                logits = self.fmodel(port, params=params).logits
                results = multiclass_log_probs(logits, port['labels'])
                self.valid_port(results["pred_ids"], results["targ_ids"])
                self.log("port", self.valid_port, on_step=False, on_epoch=True, prog_bar=True)

                # edit_inputs = batch['edit_inner']['text_input']
                # port_inputs = batch['port'][0]['text_input']
                # port_acc = results['acc'].item()
                # port_pred_ids = results['pred_ids'].cpu().numpy().tolist()
                # port_targ_ids = results['targ_ids'].cpu().numpy().tolist()

                # with open(f'{self.save_txt}', 'a') as f:
                #     f.write(f'{edit_inputs}\n{port_inputs}\n{port_acc}\npred: {port_pred_ids}\ntarget: {port_targ_ids}\n\n')
                # self.port_result.append({
                #     'edit_input': edit_inputs,
                #     'port_input': port_inputs,
                #     'port_acc': port_acc,
                #     'port_pred_ids': port_pred_ids,
                #     'port_targ_ids': port_targ_ids
                # })

    def on_validation_epoch_end(self) -> None:
        self.fmodel = None
        # with open(f'{self.save_json}', 'w') as f:
        #     json.dump(self.port_result, f, indent=2)
        return super().on_validation_epoch_end()