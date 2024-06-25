# BRIGHT
This repo provides a reference implementation of BRIGHT as described in the paper:
> Amplifying Commonsense Knowledge via Bi-directional Relation
Integrated Graph-based Contrastive Pre-training from Foundation Models

---
## Basic Usage
### Requirements
The code was tested with `python 3.10.0`, `torch 2.0.1`,`cuda 11.7`and `transformers 4.38.2` Install the dependencies via Anaconda:
```shell
# create virtual environment
conda create --name bright python=3.10

# activate environment
conda activate bright

# install cuda
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt
```
---
### Run the code
The directories **"gpt2_xl"**, **"llama2_7B"** and **"mistral_7b"** are the respective storage locations for the code and datasets of three benchmark models.

Here's an example of the code execution process using Mistral_7B:
```shell
cd ./mistral_7B

# 1. Download the mistral_7b weight file (available for online download via Hugging Face or through a local directory).
# 2. Download our trained weight files in the "new_model" directory.
# 3. Download the processed data from the "new_datas" directory.

# 4. Obtain the seed dataset
python ./dataset/build_data/generate_data.py

# 5. Generate data based on the seed dataset
python generate_new_datas.py

# 6. 进行过滤和评价
python ./post_process/post_process.py
```
---
## Datasets
The directory **"dataset"** contains the processed ConceptNet and Atomic datasets.

Datasets download link:[OneDrive](https://microsoftcrop-my.sharepoint.com/:f:/g/personal/greyhuhu_stu_my365_fit/EiWrgbkihopBtcvWtpl684kBTwhtumZImFo2ACswiVfy3g?e=6bTJsU).


The original datasets we used in the paper are come from:
- ConceptNet (Speer et al. [Conceptnet 5.5: An open multilingual graph of general knowledge](https://ojs.aaai.org/index.php/AAAI/article/view/11164). AAAI, 2017)
- ATOMIC (Sap et al. [Atomic: An atlas of machine commonsense for if-then reasoning](https://ojs.aaai.org/index.php/AAAI/article/view/4160) AAAI, 2019)


