#!/bin/bash

#SBATCH -N 2            # Number of nodes
#SBATCH -J PINN          # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m2859_g       # allocation
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:01:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpus 8
#SBATCH -o %j.out
#SBATCH -e %j.err

export MASTER_ADDR=$(hostname)

echo "jobstart $(date)";pwd

srun -n 2 python $SCRATCH/PINN/test_dist.py

echo "jobend $(date)";pwd