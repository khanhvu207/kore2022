#!/bin/bash

#SBATCH --job-name train
#SBATCH --gres gpu:4
#SBATCH --partition defq
#SBATCH --constraint A4000
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time 8:00:00

module load cuda11.5/toolkit/11.5

python3 train.py