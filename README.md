<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/VLKEB/VLKEB">
    <img src="figs/VLKEB_logo.jpg" alt="Logo" height="150">
  </a>

<h3 align="center">VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark</h3>

  [![Arxiv][arxiv-shield]][arxiv-url]
  [![Data][data-shield]][data-url]
  [![Issues][issues-shield]][issues-url]
  [![MIT License][license-shield]][license-url]

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ul>
    <li><a href="#️-about-this-project">🛠️ About This Project</a></li>
    <li><a href="#-getting-started">🚀 Getting Started</a>
        <ul>
            <li><a href="#download-data">Download Data</a></li>
            <li><a href="#environments">Environments</a></li>
            <li><a href="#pre-trained-models">Pre-trained Models</a></li>
        </ul>
    </li>
    <li><a href="#-usage">🧪 Usage</a></li>
    <li><a href="#-citation">📑 Citation</a></li>
    <li><a href="#-contact">📧 Contact</a></li>
    <li><a href="#-acknowledgments">🎉 Acknowledgments</a></li>
</ul>
</details>


<!-- ABOUT THE PROJECT -->
## 🛠️ About This Project
We construct a new Large **V**ision-**L**anguage Model **K**nowledge **E**diting **B**enchmark, **VLKEB**, and extend the Portability metric for more comprehensive evaluation. Leveraging a multi-modal knowledge graph, our image data are bound with knowledge entities. This can be further used to extract entity-related knowledge, which constitutes the base of editing data.

[![Product Name Screen Shot][product-screenshot]](https://github.com/VLKEB/VLKEB)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## 🚀 Getting Started

### Download Data
Dataset is available at [Kaggle](https://www.kaggle.com/datasets/hymanh/vlkeb-data). You can download it from site or use kaggle api:

``` bash
kaggle datasets download -d hymanh/vlkeb-data
```

The dataset is organized as follows:

```
├── VLKEB/
│   ├── VLKEB_images/           # image folder
│   │   ├── m.0104lr/           # image subfolder, entity ID
|   |   |   ├── google_15.jpg   # image file
|   |   |   ├── ...
│   │   ├── ...
|   |   
│   ├── train.json              # Train file
│   ├── eval.json               # Evaluation file, without portability test
│   ├── eval_multihop.json      # Evaluation, containing multi-hop portability
│   ├── eval_edit_onehop.json   # Evaluation, edit one-hop knowledge for portability
│   │
│   └── LICENSE.txt             # License
```

VLKEB includes a total of 8174 edits, divided into 5000 for training and 3174 for evaluation. There are 18434 images used in the Reliability, Generality, and Locality tests. The Portability test utilizes the same images as the Reliability test and comprises a total of 4819 cases. These cases are distributed among 1-hop, 2-hop, 3-hop, and 4-hop categories, with 1278, 1238, 1193, and 1110 cases, respectively.
<table>
    <thead>
        <tr>
            <th></th>
            <th><strong>All (train/eval)</strong></th>
            <th></th>
            <th><strong>Rel.</strong></th>
            <th><strong>Gen.</strong></th>
            <th><strong>Loc.</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>#Edits</strong></td>
            <td>8174 (5000/3174)</td>
            <td><strong>#Images</strong></td>
            <td>8172</td>
            <td>6627</td>
            <td>3635</td>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th></th>
            <th><strong>All (eval only)</strong></th>
            <th><strong>1-hop</strong></th>
            <th><strong>2-hop</strong></th>
            <th><strong>3-hop</strong></th>
            <th><strong>4-hop</strong></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>#Port.</strong></td>
            <td>4819</td>
            <td>1278</td>
            <td>1238</td>
            <td>1193</td>
            <td>1110</td>
        </tr>
    </tbody>
</table>


### Environments
**Conda environment:** we export the conda environment file for running the code.
We conduct experiments based on the great works in [Acknowledgments](#-acknowledgments).

```bash
# To run code of EasyEdit, use the following environment
conda env create -f envs/vlkeb_easyedit.yml

# To run code of KE, use the following environment
conda env create -f envs/vlkeb_ke.yml
```

### Pre-trained Models

To run the code, we need to download the pre-trained pytorch models and put them in proper directories.

Here we put under 'hugging_cache' folder and 'openai' folder:
```bash
# models in hugging_cache folder
hugging_cache/
├── all-MiniLM-L6-v2/
├── bert-base-uncased/
├── distilbert-base-cased/
├── Llama-2-7b-hf/
├── llava-v1.5-7b/
├── mPLUG-Owl2/
├── opt-2.7b/
├── opt-125m/
├── Qwen-7B/
├── Qwen-VL/
├── vicuna-7b/
├── vicuna-7b-v1.5/
|
├── blip2_pretrained_flant5xxl.pth
├── blip2_pretrained_opt2.7b.pth
├── eva_vit_g.pth
└── pretrained_minigpt4_7b.pth

# clip-vit model in openai folder
openai/
└── clip-vit-large-patch14-336/
``` 
Links are in the following:
<table>
    <tr>
        <td><a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">all-MiniLM-L6-v2</a></td>
        <td><a href="https://huggingface.co/google-bert/bert-base-uncased">bert-base-uncased</a></td>
        <td><a href="https://huggingface.co/distilbert/distilbert-base-cased">distilbert-base-cased</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/liuhaotian/llava-v1.5-7b">llava-v1.5-7b</a></td>
        <td><a href="https://huggingface.co/facebook/opt-2.7b">opt-2.7b</a></td>
        <td><a href="https://huggingface.co/facebook/opt-125m">opt-125m</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/Qwen/Qwen-7B">Qwen-7B</a></td>
        <td><a href="https://huggingface.co/Qwen/Qwen-VL">Qwen-VL</a></td>
        <td><a href="https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main">vicuna-7b</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">vicuna-7b-v1.5</a></td>
        <td><a href="https://huggingface.co/NousResearch/Llama-2-7b-hf">Llama-2-7b-hf</a></td>
        <td><a href="https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b">mplug-owl2-llama2-7b</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/spaces/Vision-CAIR/minigpt4/blob/main/blip2_pretrained_flant5xxl.pth">blip2_pretrained_flant5xxl.pth</a></td>
        <td><a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt2.7b.pth">blip2_pretrained_opt2.7b.pth</a></td>
        <td><a href="https://huggingface.co/spaces/Vision-CAIR/minigpt4/blob/main/prerained_minigpt4_7b.pth">prerained_minigpt4_7b.pth</a></td>
    </tr>
    <tr>
        <td><a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth">eva_vit_g.pth</a></td>
        <td><a href="https://huggingface.co/openai/clip-vit-large-patch14-336">clip-vit-large-patch14-336</a></td>
        <td></td>
    </tr>
</table>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## 🧪 Usage

Currently, we put code of different experiments in different branches. 

**BLIP2-OPT, MiniGPT-4 and LLaVA**

For the single editing experiment, you can refer to the [main](https://github.com/VLKEB/VLKEB/tree/main) branch. For the multihop and sequential editing experiment, you can refer to the [multihop_and_sequential](https://github.com/VLKEB/VLKEB/tree/multihop_and_sequential) branch. For the edit one-hop knowledge, you can refer to the [edit_onehop](https://github.com/VLKEB/VLKEB/tree/edit_onehop) branch.

For experiments of KE method, you can refer to the [main](https://github.com/VLKEB/VLKEB/tree/main) branch and get into 'KE' subfolder.

The parameters are all in [hparams](https://github.com/VLKEB/VLKEB/tree/main/hparams) folder, and detailed setting can be found in [EasyEdit](https://github.com/zjunlp/EasyEdit/blob/main/examples/MMEdit.md). Path to models and data should be properly set in config files.

To run the code, check the python file under root folder and run as the following:
```bash
# at main branch
python multimodal_edit.py [func name] [hop num]

# at main branch, KE, can use bash scripts
./train_ke.sh [GPU_ID] [MODEL_NAME]
./test_ke.sh [GPU_ID] [MODEL_NAME] [CHECKPOINT_PATH]
./test_multihop.sh [GPU_ID] [MODEL_NAME] [HOP_NUM]

# at multihop_and_sequential branch
python test_base_portability.py [func name] [hop num]
python test_multihop_portability.py [func name] [hop num]
python test_sequential_editing.py [func name]

# at edit_onehop branch
python test_edit_onehop.py [func name]
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Citation -->
## 📖 Citation

```bibtext

```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## 📧 Contact
Github (seen by all contributors) - [New Issue](https://github.com/VLKEB/VLKEB/issues/new/choose)

Han Huang - <han.huang@cripac.ia.ac.cn>

Haitian Zhong - <haitian.zhong@cripac.ia.ac.cn>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## 🎉 Acknowledgments
We would like to thank the following projects and their great works for making this project possible: [MMKG](https://github.com/mniepert/mmkb), [EasyEdit](https://github.com/zjunlp/EasyEdit), [KnowledgeEditor](https://github.com/nicola-decao/KnowledgeEditor), [LAVIS (BLIP2)](https://github.com/salesforce/LAVIS/tree/main), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen-VL](https://github.com/QwenLM/Qwen-VL), [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl).
  
We would also like to extend our gratitude to all the other projects and contributors in the open-source community whose work may not be directly listed here but has nonetheless been invaluable. Your innovations, tools, and libraries have greatly contributed to our project. We are immensely grateful for your work!

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[product-screenshot]: figs/main.png

[arxiv-shield]: https://img.shields.io/badge/Arxiv-paper-red?style=for-the-badge&logo=arxiv&logoColor=red
[arxiv-url]: https://arxiv.org/abs/2403.07350

[data-shield]: https://img.shields.io/badge/Kaggle-Dataset-blue?style=for-the-badge&logo=kaggle
[data-url]: https://www.kaggle.com/datasets/hymanh/vlkeb-data

[contributors-shield]: https://img.shields.io/github/contributors/VLKEB/VLKEB.svg?style=for-the-badge
[contributors-url]: https://github.com/VLKEB/VLKEB/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/VLKEB/VLKEB.svg?style=for-the-badge
[forks-url]: https://github.com/VLKEB/VLKEB/network/members

[stars-shield]: https://img.shields.io/github/stars/VLKEB/VLKEB.svg?style=for-the-badge
[stars-url]: https://github.com/VLKEB/VLKEB/stargazers

[issues-shield]: https://img.shields.io/github/issues/VLKEB/VLKEB.svg?style=for-the-badge
[issues-url]: https://github.com/VLKEB/VLKEB/issues

[license-shield]: https://img.shields.io/github/license/VLKEB/VLKEB.svg?style=for-the-badge
[license-url]: https://github.com/VLKEB/VLKEB/blob/main/LICENSE