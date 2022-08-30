#!/bin/bash
#SBATCH -c12
#SBATCH --gres=gpu:turing:1
#SBATCH -p gpu
module load anaconda3 cuda/11 cudnn/10-74
source activate pcdan_iros2022
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES


python -u eval_seq-ss.py