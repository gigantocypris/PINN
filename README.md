# PINN

References:

Learning differentiable solvers for systems with hard constraints
https://arxiv.org/abs/2207.08675

## Setup on NERSC

cd $SCRATCH
git clone https://github.com/gigantocypris/PINN.git

module load python
conda create -n PINN python=3.8 -y
conda activate PINN

salloc -N 1 --time=60 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=8 --cpus-per-task=16

conda install pytorch==2.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

> cd $SCRATCH/PINN
> python test_pytorch_install.py

conda install h5py
python -m pip install --upgrade pip

python -m pip install scipy
python -m pip install pandas
python -m pip install matplotlib
python -m pip install pytest
python -m pip install pillow
python -m pip install tensorboard
python -m pip install tensorboardX

python -m pip install tfrecord
python -m pip install dxchange
python -m pip install six
python -m pip install tifffile
python -m pip install -U scikit-learn
python -m pip install scikit-optimize





