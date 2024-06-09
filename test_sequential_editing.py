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

def test_MiniGPT4_FT():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_MiniGPT4_FT_VIS():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_qformer.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_MiniGPT4_MEND():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_MiniGPT4_SERAC():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)


####################### BLIP2 ##########################
def test_Blip2OPT_FT():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_Blip2OPT_FT_VIS():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2_qformer.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_Blip2OPT_MEND():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_Blip2OPT_SERAC():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

####################### LLAVA ##########################

def test_LLaVA_FT():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_LLaVA_FT_VIS():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_mmproj.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_LLaVA_MEND():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/llava.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_LLaVA_SERAC():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/llava.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)


if __name__ == "__main__":
    function_name = sys.argv[1]
    hop = 1
    os.makedirs('results/results_sequencial', exist_ok=True)

    eval_json_path = 'datasets/eval_multihop.json'
    if function_name not in globals() or not callable(globals()[function_name]):
        print(f"Error: Function '{function_name}' does not exist.")
        sys.exit(1)
    for gap_num in [10, 20, 50, 100]:
        globals()[function_name]()
    
    
