#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J PINN          # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m2859_g       # allocation
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:05:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpus 4
#SBATCH -o %j.out
#SBATCH -e %j.err

export MASTER_ADDR=$(hostname)
export BATCH_SIZE=512
export SCRATCH_FOLDER=$SCRATCH/output_PINN/$SLURM_JOB_ID
mkdir -p $SCRATCH_FOLDER; cd $SCRATCH_FOLDER

echo "jobstart $(date)";pwd

srun -n 4 -c 32 python $SCRATCH/PINN/main.py --upc --2d --dist --bs $BATCH_SIZE --epochs 5

echo "jobend $(date)";pwd