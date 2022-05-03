#!/bin/bash

#SBATCH --job-name tensors
#SBATCH --gres gpu:1
#SBATCH --partition defq
#SBATCH --constraint A4000
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --time 8:00:00

module load cuda11.5/toolkit/11.5

python3 make_tensors.py