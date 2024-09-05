import torch.nn as nn
import torch
from copy import deepcopy

from ..losses import masked_log_probs
from ..utils import _logits, shift_targets


class EditableModel(nn.Module):
    def __init__(self, model, config, model_constructor):
        super().__init__()

        self.model = model
        self.config = deepcopy(config)
        self.model_constructor = model_constructor

        def _edit_loss_fn(config, pred, targ):
            return masked_log_probs(config, pred, targ, shift=True)

        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = masked_log_probs

    def edit(self, batch, condition=None, detach_history=False):
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        if self.config.model_name == "qwen-vl":
            return _logits(self.model(inputs[0]['inputs']))
        elif self.config.model_name == "owl-2":
            input_ids, image = inputs[0]['input_ids'], inputs[0]['image']
            return _logits(self.model(input_ids.to(self.config.device), 
                                         images=image.to(self.config.device, dtype=torch.float16)))
        else:
            return _logits(self.model(*inputs, **kwargs))

    def outer_parameters(self):
        return self.parameters()

    def base_loss(self, input_ids, attention_masks, label_ids):
        pass
