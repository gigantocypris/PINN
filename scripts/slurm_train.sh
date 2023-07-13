#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J PINN          # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m2859_g       # allocation
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 07:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpus 4
#SBATCH -o %j.out
#SBATCH -e %j.err

export MASTER_ADDR=$(hostname)
export SCRATCH_FOLDER=$SCRATCH/output_PINN/$SLURM_JOB_ID
mkdir -p $SCRATCH_FOLDER; cd $SCRATCH_FOLDER

echo "jobstart $(date)";pwd

python $SCRATCH/PINN/main.py --2d --dist --epochs 30000 --bs 872356
# python $SCRATCH/PINN/main.py --upc --2d --dist --bs 8192 --epochs 500

echo "jobend $(date)";pwd