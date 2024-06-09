import os
import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, FTMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import sys
from datetime import datetime



####################### MiniGPT4 ##########################

def test_MiniGPT4():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_onehop.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_knowledge(log=True)


####################### BLIP2 ##########################
def test_Blip2OPT():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2_onehop.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_knowledge(log=True)

####################### LLAVA ##########################

def test_LLaVA():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_onehop.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_knowledge(log=True)


if __name__ == "__main__":
    function_name = sys.argv[1]
    hop = sys.argv[2]
    os.makedirs('results/results_base_port', exist_ok=True)

    eval_json_path = 'datasets/eval_multihop.json'
    if function_name not in globals() or not callable(globals()[function_name]):
        print(f"Error: Function '{function_name}' does not exist.")
        sys.exit(1)
    globals()[function_name]()
    
    
