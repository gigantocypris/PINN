#!/bin/bash

#SBATCH -N 0.25          # Number of nodes
#SBATCH -J PINN          # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m2859_g       # allocation
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 00:05:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=2
#SBATCH --gpus 128
#SBATCH -o %j.out
#SBATCH -e %j.err

export PERL_NDEV=1  # number GPU per node
export PANDA=$SCRATCH/ferredoxin_sim/9713113/out/preds_for_hopper.pkl
export IBV_FORK_SAFE=1
export RDMAV_HUGEPAGES_SAFE=1
export DIFFBRAGG_USE_CUDA=1

echo "jobstart $(date)";pwd


srun -n 256 -c 16 \
ens.hopper $PANDA $MODULES/exafel_project/kpp-sim/ens_hopper.phil \
--outdir $SCRATCH/ferredoxin_sim/preimport --maxSigma 3 --saveFreq 10  --preImport --refl predictions

srun -n 256 -c 16 \
ens.hopper $SCRATCH/ferredoxin_sim/preimport/preImport_for_ensemble.pkl $MODULES/exafel_project/kpp-sim/ens_hopper.phil \
--outdir global --maxSigma 3 --saveFreq 10 --refl ens.hopper.imported \
--cmdlinePhil fix.Nabc=True fix.ucell=True fix.RotXYZ=True fix.Fhkl=False fix.G=False sigmas.G=1e-2

echo "jobend $(date)";pwd