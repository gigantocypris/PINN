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


# Function to list GPUs on each node
function list_gpus {
    node=$1
    echo "Listing GPUs on node $node:"
    srun --nodes=1 --ntasks=1 --gpus-per-task=1 -w $node nvidia-smi --list-gpus
    echo ""
}

# Get the allocated nodes
nodes=$(scontrol show hostname $SLURM_JOB_NODELIST)

# Loop over each node and list GPUs
for node in $nodes; do
    list_gpus $node
done

# Print the GPU devices
# nvidia-smi --list-gpus
