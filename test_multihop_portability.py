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


def print_result(metrics, save_path=None):
    # rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
    # rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
    # rephrase_image_acc = mean([m['post']['rephrase_image_acc'].item() for m in metrics])
    # locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
    # locality_image_acc = mean([m['post']['locality_image_acc'].item() for m in metrics])
    # print(f'rewrite_acc: {rewrite_acc}')
    # print(f'rephrase_acc: {rephrase_acc}')
    # print(f'rephrase_image_acc: {rephrase_image_acc}')
    # print(f'locality_acc: {locality_acc}')
    # print(f'locality_image_acc: {locality_image_acc}')

    ### portability
    portability_acc = mean([m['post']['portability_acc'].item() for m in metrics if 'portability_acc' in m['post']])
    print(f'portability_acc: {portability_acc}')

    if save_path is not None:
        with open(save_path, 'w') as f:
            # f.write(f'rewrite_acc: {rewrite_acc}\n')
            # f.write(f'rephrase_acc: {rephrase_acc}\n')
            # f.write(f'rephrase_image_acc: {rephrase_image_acc}\n')
            # f.write(f'locality_acc: {locality_acc}\n')
            # f.write(f'locality_image_acc: {locality_image_acc}\n')

            #### portability
            f.write(f'portability_acc: {portability_acc}\n')


def Generate_Embedding_for_IKE():
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    train_ds = CaptionDataset(train_json_path, config=hparams, no_image=True)
    ## Generate embedding files for IKE
    sentence_model = SentenceTransformer(hparams.sentence_model_name, device=f'cuda:{hparams.device}')
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)


####################### MiniGPT4 ##########################
def train_MEND_MiniGPT4_Caption():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = CaptionDataset(train_json_path, config=hparams)
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    


def test_MEND_MiniGPT4_Caption():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()


def train_SERAC_MiniGPT4_Caption():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = CaptionDataset(train_json_path, config=hparams)
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()


def test_SERAC_MiniGPT4_Caption():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()

def test_FT_MiniGPT4():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_MiniGPT4_Qformer():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_qformer.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_IKE_MiniGPT4():
    cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    save_path = f'results/IKE/{cur_time}_{hparams.model_name}_results_port_hop{hop}.txt'
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        cur_time=cur_time        
    )
    
    print_result(metrics, save_path)


####################### BLIP2 ##########################
def train_MEND_Blip2OPT_Caption():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    train_ds = CaptionDataset(train_json_path, config=hparams)
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
    

def test_MEND_Blip2OPT_Caption():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()    

    
def train_SERAC_Blip2OPT_Caption():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    train_ds = CaptionDataset(train_json_path, config=hparams)
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()


def test_SERAC_Blip2OPT_Caption():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()

def test_FT_Blip2OPT():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_Blip2OPT_QFormer():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2_qformer.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_IKE_Blip2OPT():
    cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    save_path = f'results/IKE/{cur_time}_{hparams.model_name}_results_port_hop{hop}.txt'
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        cur_time=cur_time        
    )
    
    print_result(metrics, save_path)


####################### LLAVA ##########################
def train_MEND_LLaVA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/llava.yaml')
    train_ds = CaptionDataset(train_json_path, config=hparams)
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def test_MEND_LLaVA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/llava.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def train_SERAC_LLaVA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/llava.yaml')
    train_ds = CaptionDataset(train_json_path, config=hparams)
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()    

def test_SERAC_LLaVA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/llava.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()   

def test_FT_LLaVA():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_FT_LLaVA_mmproj():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_mmproj.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_IKE_LLaVA():
    cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/llava.yaml')
    save_path = f'results/IKE/{cur_time}_{hparams.model_name}_results_port_hop{hop}.txt'
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True,
        cur_time=cur_time        
    )
    
    print_result(metrics, save_path)

if __name__ == "__main__":
    function_name = sys.argv[1]
    hop = sys.argv[2]

    train_json_path = 'datasets/train.json'
    eval_json_path = 'datasets/eval_multihop.json'
    os.makedirs('results/results_multihop', exist_ok=True)

    if function_name not in globals() or not callable(globals()[function_name]):
        print(f"Error: Function '{function_name}' does not exist.")
        sys.exit(1)
    globals()[function_name]()
