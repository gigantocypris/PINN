#!/bin/bash

#SBATCH -N 0.25          # Number of nodes
#SBATCH -J PINN          # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m2859_g       # allocation
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:05:00
#SBATCH --gpus 1
#SBATCH -o %j.out
#SBATCH -e %j.err

export SCRATCH_FOLDER=$SCRATCH/output_PINN/$SLURM_JOB_ID
mkdir -p $SCRATCH_FOLDER; cd $SCRATCH_FOLDER

echo "jobstart $(date)";pwd

srun -n 1 -c 32 python $SCRATCH/PINN/main.py

echo "jobend $(date)";pwd