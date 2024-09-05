import logging
import torch.nn as nn
from transformers import GPT2Tokenizer, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM


LOG = logging.getLogger(__name__)


def get_model(model_name, device=None):
    model_path = "../hugging_cache"
    if model_name == "blip2":
        from .blip2_models.blip2_opt import Blip2OPT
        
        model = Blip2OPT(
            vit_model="eva_clip_g",
            img_size=364,
            use_grad_checkpoint=True,
            vit_precision="fp32",
            freeze_vit=True,
            freeze_qformer=True,
            opt_model=f'{model_path}/opt-2.7b',
            state_dict_file=f'{model_path}/eva_vit_g.pth',
            qformer_name_or_path=f'{model_path}/bert-base-uncased',
            qformer_checkpoint=f'{model_path}/blip2_pretrained_opt2.7b.pth'
        )
        tokenizer = GPT2Tokenizer.from_pretrained(f'{model_path}/opt-2.7b')

    elif model_name == "minigpt4":
        from .blip2_models.mini_gpt4 import MiniGPT4

        model = MiniGPT4(
            vit_model="eva_clip_g",
            qformer_checkpoint=f'{model_path}/blip2_pretrained_flant5xxl.pth',
            img_size=364,
            use_grad_checkpoint=True,
            vit_precision="fp32",
            freeze_vit=True,
            freeze_qformer=True,
            llama_model=f'{model_path}/vicuna-7b',
            state_dict_file=f'{model_path}/eva_vit_g.pth',
            qformer_name_or_path=f'{model_path}/bert-base-uncased',
            pretrained_ckpt=f'{model_path}/pretrained_minigpt4_7b.pth',
        )
        tokenizer = LlamaTokenizer.from_pretrained(f'{model_path}/vicuna-7b')

    elif model_name == "llava":
        from .llava.model.builder import load_pretrained_model
        model = load_pretrained_model(model_path=f'{model_path}/llava-v1.5-7b')
        tokenizer = LlamaTokenizer.from_pretrained(f'{model_path}/llava-v1.5-7b')

    elif "qwen-vl" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token='<|endoftext|>')
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        )

    elif "owl2" in model_name.lower():
        from .mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model
        tokenizer , model, _, _ = load_pretrained_model(model_name, None, 'mplug_owl2', load_8bit=False, load_4bit=False)
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # print model parameters
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape} {param.dtype}")
    

    n_reset = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
            n_reset += 1

    LOG.info(f"Set {n_reset} dropout modules to p={0.0}")

    return model, tokenizer
