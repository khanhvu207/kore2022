#!/bin/bash

#SBATCH --job-name train
#SBATCH --gres gpu:1
#SBATCH --partition fatq
#SBATCH --constraint A100
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time 12:00:00

module load cuda11.5/toolkit/11.5

python3 train.py --lr=3e-4\
    --weight_decay=1e-4\
    --warmup_steps=1000\
    --num_epochs=5\
    --num_gpus=1\
    --debug=False