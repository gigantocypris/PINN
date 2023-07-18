#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J PINN          # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m2859_g       # allocation
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:50:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpus 4
#SBATCH -o %j.out
#SBATCH -e %j.err

export MASTER_ADDR=$(hostname)
export SCRATCH_FOLDER=$SCRATCH/output_PINN/$SLURM_JOB_ID
mkdir -p $SCRATCH_FOLDER; cd $SCRATCH_FOLDER

echo "jobstart $(date)";pwd
echo SLURM_NTASKS

python $SCRATCH/PINN/main.py --2d --epochs 10000 --bs 160000 -j 0.025 --train_x_step 0.05 0.05

echo "jobend $(date)";pwd