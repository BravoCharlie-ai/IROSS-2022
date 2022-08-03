#!/bin/bash
#SBATCH -c8
#SBATCH --gres=gpu:ampere:1
#SBATCH -p gpu
module load anaconda3 cuda/11 cudnn/11
source activate mmmot_laptop_copy
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

python -u eval_seq.py