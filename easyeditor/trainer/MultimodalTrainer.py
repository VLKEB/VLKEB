from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)
from tqdm import tqdm
from transformers import AutoTokenizer

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"], detach_history=True)
        edit_time = time.time() - start

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0
        info_dict = {}

        ################ portability #################
        if batch['port'] is not None:
            assert len(batch['port']) == 1 and len(batch['edit_port_2']) == 1, "edit twice only support one portability edit"

            edited_model, model_info = edited_model.edit(batch["edit_port_2"][0], batch["cond"])
            port = batch['port'][0]
            with torch.no_grad():
                port_outputs = edited_model(port)
                port_labels = port["labels"]
                if not isinstance(port_outputs, torch.Tensor):
                    port_logits = port_outputs.logits
                else:
                    port_logits = port_outputs
                if port_logits.shape[1] > port_labels.shape[1]:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                else:
                    port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                port_acc = port_dict["acc"].item()
            info_dict['port/acc'] = port_acc
        ################ portability #################
        
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        if 'port/acc_val' in stats:
            LOG.info(f"step {prog} port_acc: {stats['port/acc_val']:<12.5f} it_time: {elapsed:.4f}")

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            if (log and (val_step) % self.config.log_interval == 0):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )
           
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats