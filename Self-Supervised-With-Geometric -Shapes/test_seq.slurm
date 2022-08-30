#!/bin/bash
#SBATCH -c12
#SBATCH --gres=gpu:1
#SBATCH -p gpu
module load anaconda3 cuda/11 cudnn/10-74
source activate mmmot_laptop_copy
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python -u test_seq.py