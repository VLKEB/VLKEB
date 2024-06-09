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
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0
        info_dict = {}

        ################ portability #################
        if batch['port'] is not None:
            port_acc = 0
            assert len(batch['port']) == 1, "batch['port'] should have only one element"
            for port in batch['port']:
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
                    port_acc += port_dict["acc"].item()
                    info_dict['grad/port_pred_ids'] = port_dict['pred_ids']
                    info_dict['grad/port_targ_ids'] = port_dict['targ_ids']
            port_acc /= len(batch['port'])
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
        #################################################################
        # inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        # outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        # image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        # loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        # loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"

        # LOG.info(
        #   f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}"
        # )
        #################################################################
        if 'port/acc_val' in stats:
            LOG.info(f"step {prog} port_acc: {stats['port/acc_val']:<12.5f} it_time: {elapsed:.4f}")
       
        if 'knowledge/acc_val' in stats:
            LOG.info(f"step {prog} knowledge_acc: {stats['knowledge/acc_val']:<12.5f} it_time: {elapsed:.4f}")

    def validate(self, steps=None, log: bool = False, result_name: str = None):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        if result_name is not None:
            port_result = []
        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            
            if (log and (val_step) % self.config.log_interval == 0):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )
           
            if batch['port'] is None:
                continue
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict) 

            # append write to txt file info_dict['port/acc']
            if result_name is not None:
                edit_inputs = batch['edit_inner']['text_input']
                port_inputs = batch['port'][0]['text_input']
                port_acc = info_dict['port/acc']
                port_pred_ids = info_dict['grad/port_pred_ids'].cpu().numpy()
                port_targ_ids = info_dict['grad/port_targ_ids'].cpu().numpy()
                # with open(f'results/results_multihop/{result_name}_port_hop{self.val_set.hop}.txt', 'a') as f:
                #     f.write(f'{edit_inputs}\n{port_inputs}\n{port_acc}\npred: {port_pred_ids}\ntarget: {port_targ_ids}\n\n')
                port_result.append({
                    'edit_input': edit_inputs,
                    'port_input': port_inputs,
                    'port_acc': port_acc,
                    'port_pred_ids': port_pred_ids.tolist(),
                    'port_targ_ids': port_targ_ids.tolist()
                })
        
        if result_name is not None:
            with open(f'results/results_multihop/{result_name}_port_hop{self.val_set.hop}.json', 'w') as f:
                json.dump(port_result, f, indent=2)

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats
    
    def knowledge_qa(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0
        info_dict = {}

        assert batch['port'] is not None, "portability edit must be provided"
        assert len(batch['port']) == 1, "batch['port'] should have only one element"

        knowledge = batch['port'][0]
        with torch.no_grad():
            knowledge_outputs = self.model(knowledge)
            knowledge_labels = knowledge["labels"]
            if not isinstance(knowledge_outputs, torch.Tensor):
                knowledge_logits = knowledge_outputs.logits
            else:
                knowledge_logits = knowledge_outputs
            if knowledge_logits.shape[1] > knowledge_labels.shape[1]:
                knowledge_dict = self.model.edit_loss_fn(self.config, knowledge_logits, knowledge_labels)
            else:
                knowledge_dict = self.model.edit_loss_fn(self.config, knowledge_logits, knowledge_labels[:, -knowledge_logits.shape[1]-1:])
            knowledge_acc = knowledge_dict["acc"].item()
        info_dict['knowledge/acc'] = knowledge_acc
        
        info_dict = {**info_dict, **{}}

        return l_total, l_edit, l_loc, l_base, info_dict
    
    def test_knowledge(self, steps=None, log: bool = False):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.eval()

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

            _, _, _, _, info_dict = self.knowledge_qa(batch, training=False)
            averager.add(info_dict) 

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps


        results_path = f"results/results_base_port/{cur_time}_{self.config.model_name}_port{self.val_set.hop}_questiontest.json"

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats

    
    def _inline_seq_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"
        port_acc = f"{stats['port/acc_val']:<12.5f}"
        LOG.info(
          f"Step {prog} outer_acc: {outer_acc} image_acc: {image_acc} inner_acc: {inner_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc}, port_acc: {port_acc}"
        )


    def test_sequencial_step(self, batch, edited_model, base_logits, base_image_logits):
        info_dict = {}

        ##############################################################################
        with torch.no_grad():
            # inner
            inner_edit_outputs = edited_model(batch["edit_inner"])
            inner_batch_labels = batch["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
            else:
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
            del inner_edit_outputs, inner_edit_logits
            torch.cuda.empty_cache()

            # text rephrase
            post_edit_outputs = edited_model(batch["edit_outer"])
            post_batch_labels = batch["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
            else:
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
            del post_edit_outputs, post_edit_logits
            torch.cuda.empty_cache()

            # image rephrase
            post_image_edit_outputs = edited_model(batch["edit_outer_image"])
            post_image_batch_labels = batch["edit_outer_image"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
            else:
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
            del post_image_edit_outputs, post_image_edit_logits
            torch.cuda.empty_cache()

            # text loc
            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
            else:
                post_base_logits = post_base_outputs
            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
            del post_base_outputs, post_base_logits
            torch.cuda.empty_cache()

            # image loc
            post_image_base_outputs = edited_model(batch["loc_image"])
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
            else:
                post_image_base_logits = post_image_base_outputs
            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
            del post_image_base_outputs, post_image_base_logits
            torch.cuda.empty_cache()

        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        ##############################################################################

        ################ portability #################
        assert len(batch['port']) == 1, "batch['port'] exist and have only one element"
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

        return info_dict

    def test_sequencial(self, log: bool = False, test_num=200, gap_num=0):
        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.model.train(True)

        steps = test_num + gap_num
        if log:
            LOG.info(f"Beginning evaluation for {test_num} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        val_data_store = []
        base_logits_store = []
        base_image_logits_store = []
        pbar = tqdm(total=test_num, desc=f"Prepare", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            if val_step < test_num:
                val_data_store.append(batch)
                with torch.no_grad():
                    base_outputs = self.model(batch["loc"])
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs
                    base_logits_store.append(base_logits.clone().detach())
                        
                    base_image_outputs = self.model(batch["loc_image"])
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    base_image_logits_store.append(base_image_logits.clone().detach())
                pbar.update(1)
            else:
                break
        pbar.close()

        edited_model = self.model
        pbar = tqdm(total=gap_num+test_num, desc=f"Test Gap {gap_num}", ncols=100)
        for val_step, batch in enumerate(self.val_loader):
            edited_model, _ = edited_model.edit(batch["edit_inner"], batch["cond"], detach_history=True)
            if val_step >= gap_num:
                stored_batch = val_data_store.pop(0)
                stored_base_logits = base_logits_store.pop(0)
                stored_base_image_logits = base_image_logits_store.pop(0)
                info_dict = self.test_sequencial_step(stored_batch, edited_model, stored_base_logits, stored_base_image_logits)
                averager.add(info_dict)

            if (log and val_step >= gap_num and (val_step) % self.config.log_interval == 0):
                self._inline_seq_log(
                    val_step, averager.average(), start_time, steps
                )
            pbar.update(1)

            if len(val_data_store) == 0:
                break
        pbar.close()

        if log:
            self._inline_seq_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        results_path = f"results/results_sequencial/{cur_time}_{self.config.alg}_{self.config.model_name}_port{self.val_set.hop}_seqgap{gap_num}.json"

        with open(results_path, "w") as f:
            json.dump(
                {"results": stats}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        return stats