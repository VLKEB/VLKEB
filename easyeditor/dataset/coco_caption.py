"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from tqdm import tqdm
from copy import deepcopy

class CaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError("unknown model class")

        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: "

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "":
                continue
            if hop and 'port_new' not in record.keys():
                continue
            
            image_path = os.path.join(self.vis_root, record["image"])
            rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            locality_image_path = os.path.join(self.vis_root, record['m_loc'])
            
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                # 'image': image,
                # 'image_rephrase': rephrase_image,
                'image': image_path,
                'image_rephrase': rephrase_image_path,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }
            
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            
            # item['multimodal_locality_image'] = locality_image
            item['multimodal_locality_image'] = locality_image_path

            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']

            if hop and 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question']
                        port_a = ports['Q&A']['Answer']
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break
                
                if not find_hop:
                    continue
            data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image

    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])        
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0)
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # edit_outer
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0)
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        
        # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # m_loc
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0)
        loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        edit_ports = None
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                port['image'] = torch.stack(image, dim=0)
                port['text_input'] = [' '.join([port_q, port_a])]
                port['labels'] = [port_a]
                port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt",)["input_ids"]
                edit_ports.append(port)

        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            'port': edit_ports,
            "cond": cond
        }
        return dict_to(batch, self.config.device)
