#!/usr/bin/env python3
"""
Stable Audio Open Finetuning Script

Converted from a Jupyter notebook. Shell commands (prefixed with "$ ") should be
run in a terminal before/after the Python cells. This script documents the full
finetuning workflow: environment setup, model download, dataset preparation,
training, and checkpoint upload.
"""

# $ source activate pytorch_p310

# $ nvidia-smi

# $ pip install numpy==1.24.3 PyWavelets==1.4.1

# $ conda install -c conda-forge h5py soxr-python -y

# $ git clone https://github.com/Stability-AI/stable-audio-tools.git
# $ cd /path/to/stable-audio-tools
# $ pip install -e .
# $ cd ..

# $ conda install -c conda-forge ffmpeg -y

# $ pip install gdown

from huggingface_hub import hf_hub_download
from huggingface_hub import notebook_login

notebook_login()

hf_hub_download(repo_id="stabilityai/stable-audio-open-1.0", filename="model.ckpt", local_dir="./")

# $ gdown 1BK3lPVGDmAOBOh0Fz4u-g2m6dRAVwoJR

# $ yum install -y unzip

# $ unzip "Archive.zip" -d "/workspace/dataset"

#%cd dataset
#!rm -rf __MACOSX
#%cd ..

# $ wandb login YOUR_WANDB_KEY

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # MUST be set BEFORE running train.py

# $ cd /path/to/stable-audio-tools

# $ pip uninstall numpy -y

# $ pip install numpy==1.24.3 scipy==1.11.4 --force-reinstall

# $ pip install -U deepspeed

# $ # Clear any cached memory first
import torch
torch.cuda.empty_cache()

# $ python3 train.py \
#     --dataset-config /path/to/configs/dataset_config.json \
#     --model-config /path/to/configs/model_config.json \
#     --name run_01 \
#     --save-dir /path/to/checkpoints \
#     --pretrained-ckpt-path /path/to/model.ckpt \
#     --checkpoint-every 1000 \
#     --batch-size 2 \
#     --strategy deepspeed \
#     --precision 16

# $ aws s3 sync /path/to/checkpoints/run_01/ \
#     s3://training-checkpoints/stable-audio-checkpoints/run_01/
