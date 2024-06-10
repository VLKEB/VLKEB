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


def print_result(metrics, save_path=None):
    ### portability
    portability_acc = mean([m['post']['portability_acc'].item() for m in metrics if 'portability_acc' in m['post']])
    print(f'portability_acc: {portability_acc}')

    if save_path is not None:
        with open(save_path, 'w') as f:
            #### portability
            f.write(f'portability_acc: {portability_acc}\n')


def Generate_Embedding_for_IKE():
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    train_ds = CaptionDataset(train_json_path, config=hparams, no_image=True)
    ## Generate embedding files for IKE
    sentence_model = SentenceTransformer(hparams.sentence_model_name, device=f'cuda:{hparams.device}')
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)


####################### MiniGPT4 ##########################
def test_MEND_MiniGPT4():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_SERAC_MiniGPT4():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_MiniGPT4():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_MiniGPT4_Qformer():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_qformer.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams, edit_twice_img=True)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


####################### BLIP2 ##########################
def test_MEND_Blip2OPT():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()    


def test_SERAC_Blip2OPT():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_Blip2OPT():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_Blip2OPT_QFormer():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2_qformer.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams, edit_twice_img=True)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


####################### LLAVA ##########################

def test_MEND_LLaVA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/llava.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()    


def test_SERAC_LLaVA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/llava.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()   

def test_FT_LLaVA():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_LLaVA_mmproj():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_mmproj.yaml')
    eval_ds = CaptionDataset(eval_path, config=hparams, edit_twice_img=True)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()



if __name__ == "__main__":
    function_name = sys.argv[1]

    train_json_path = 'datasets/train.json'
    eval_path = 'datasets/eval_edit_onehop.json'
    
    if function_name not in globals() or not callable(globals()[function_name]):
        print(f"Error: Function '{function_name}' does not exist.")
        sys.exit(1)
    globals()[function_name]()
