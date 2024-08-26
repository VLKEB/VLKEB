#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoTokenizer
from transformers.utils import ModelOutput
from dataclasses import dataclass

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)

@dataclass
class LLaVAOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.IntTensor = None
    attention_mask: torch.IntTensor = None


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        print("***********vocab_size", config.vocab_size)
        print("************** lm head", self.lm_head.weight.shape)
        # Initialize weights and apply final processing
        self.post_init()
        self.tokenizer = AutoTokenizer.from_pretrained('hugging_cache/llava-onevision-qwen2-7b-ov', use_fast=False)


    def get_model(self):
        return self.model

    def prepare_inputs_from_batch(self, samples):
        text = [t for t in samples["text_input"]]

        input_tokens = self.tokenizer(text, padding=True, return_tensors='pt').to(self.device)
        input_ids = input_tokens.input_ids
        attention_mask = input_tokens.attention_mask
        
        if samples['image'] is not None:
            image = samples["image"].to(self.dtype) # bsz, 3, image_size, image_size
            image_token_ids = torch.ones((input_ids.shape[0]), dtype=input_ids.dtype, device=self.device).fill_(IMAGE_TOKEN_INDEX)
            input_ids = torch.cat((input_ids[:, :1], image_token_ids.unsqueeze(1), input_ids[:, 1:]), dim=1)

            image_att_mask = torch.ones((input_ids.shape[0]), dtype=input_ids.dtype, device=self.device)
            attention_mask = torch.cat((attention_mask[:, :1], image_att_mask.unsqueeze(1), attention_mask[:, 1:]), dim=1)
            
            targets = input_ids.masked_fill(input_ids==self.tokenizer.pad_token_id, IGNORE_INDEX)
            if samples['prompts_len']:
                for i, prompt_len in enumerate(samples['prompts_len']):
                    targets[i, :prompt_len+1] = IGNORE_INDEX
        else:
            image = None
            targets = input_ids.masked_fill(input_ids==self.tokenizer.pad_token_id, IGNORE_INDEX)
            if samples['prompts_len']:
                for i, prompt_len in enumerate(samples['prompts_len']):
                    targets[i, :prompt_len] = IGNORE_INDEX

        return self.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            attention_mask,
            None,
            targets,
            image
        )
    
    def forward(self, samples):
        
        (   input_ids,
            _,
            attention_mask,
            _,
            inputs_embeds,
            targets
        ) = self.prepare_inputs_from_batch(samples)

        # outputs = super().forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=None,
        #         past_key_values=None,
        #         inputs_embeds=inputs_embeds,
        #         output_hidden_states=True,
        #         labels=targets,
        #         return_dict=True,
        #     )
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        print(self.lm_head.weight.shape)

        # print("output.logits shape", outputs.logits.shape)

        if torch.isnan(logits).any():
            print("LLaVA-OV logits has nan!!!!!!!!!!!!!!!")

        # return LLaVAOutput(
        #     loss=outputs.loss,
        #     logits=outputs.logits,
        #     labels=targets,
        #     attention_mask=attention_mask
        # )
        return LLaVAOutput(
            loss=None,
            logits=logits,
            labels=targets,
            attention_mask=attention_mask
        )

    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     images: Optional[torch.FloatTensor] = None,
    #     image_sizes: Optional[List[List[int]]] = None,
    #     return_dict: Optional[bool] = None,
    #     modalities: Optional[List[str]] = ["image"],
    #     dpo_forward: Optional[bool] = False,
    #     cache_position=None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:

    #     if inputs_embeds is None:
    #         (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

    #     if dpo_forward:
    #         outputs = self.model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    #         hidden_states = outputs[0]
    #         logits = self.lm_head(hidden_states)
    #         return logits, labels

    #     else:
    #         return super().forward(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             labels=labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
