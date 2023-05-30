<br/>
<p align="center"><img src="images/logo.png" width=500 /></p>

----
![Crates.io](https://img.shields.io/crates/l/Ap?color=black)

# PINN

References:

Learning differentiable solvers for systems with hard constraints
https://arxiv.org/abs/2207.08675

## Setup on MacBook Pro

Prerequisite: install miniconda or Anaconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

Open the Terminal:
conda create -n PINN python=3.8 -y
conda activate PINN


## Setup on NERSC

cd $SCRATCH
git clone https://github.com/gigantocypris/PINN.git

module load python
conda create -n PINN python=3.8 -y
conda activate PINN

salloc -N 1 --time=60 -C gpu -A m3562_g --qos=interactive --ntasks-per-gpu=8 --cpus-per-task=4

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

## Resources


PINN:
Main paper with data:
https://iopscience.iop.org/article/10.1088/0266-5611/21/6/S09

Efficient inversion of multiple-scattering model for optical diffraction tomography (SEAGLE extension)
https://opg.optica.org/oe/fulltext.cfm?uri=oe-25-18-21786&id=371123


A Fast Algorithm of Cross-Correlated Contrast Source Inversion in Homogeneous Background Media
https://ieeexplore.ieee.org/document/10044704
https://github.com/TUDsun/CC-CSI
https://github.com/TUDsun/GMMV-LIM

T-Matrix Backprojection Imaging for Scalar
and Vector Electromagnetic Waves
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10006692
https://github.com/nasa-jpl/Waveport


Deep Learning-Based Inverse Scattering With
Structural Similarity Loss Functions
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9220912
https://github.com/VainF/pytorch-msssim

Closing the Gap of Simulation to Reality in
Electromagnetic Imaging of Brain Strokes
via Deep Neural Networks
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9274540
https://github.com/thisismygitrepo/emi_da


## PINN notes

Notes for creating the PINN
https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html

Creating pytorch 2.0 conda env on local laptop:

conda create -n pytorch2 python=3.9
conda install pytorch torchvision torchaudio -c pytorch
python -m pip install -U pip
python -m pip install -U matplotlib
python -m pip install scipy

Using phys-ml environment on mac brick
Implementing least squares layer:

https://github.com/cvxgrp/cvxpylayers/issues/2:

@luzai Also, I will add that for a least squares problem, you are probably much better off implementing the solution (and its derivative) directly, since both have closed-form expressions. You should use our package for more exotic problems, for example, non-negative least squares.

For a direct implementation of least squares, you could use a least squares PyTorch layer I wrote with analytical derivatives: https://github.com/sbarratt/lsat/blob/master/lstsq.py. (Simply copy DenseLeastSquares into your code.)

(do timing comparison with PyTorch layer)

NERSC notes:
module avail pytorch
module load pytorch/1.13.1

https://github.com/NERSC/pytorch-examples

Use DistributedDataParallel (DDP):
https://pytorch.org/tutorials/beginner/dist_overview.html


useful srun reference: https://slurm.schedmd.com/srun.html
notes on this script: https://github.com/NERSC/pytorch-examples/blob/main/scripts/train_perlmutter.sh

$@ is all of the parameters passed to the script.

For instance, if you call ./someScript.sh foo bar then $@ will be equal to foo bar.


-l, --label
Prepend task number to lines of stdout/err. The --label option will prepend lines of output with the remote task id. This option applies to step allocations.

-u, --unbuffered
By default, the connection between slurmstepd and the user-launched application is over a pipe. The stdio output written by the application is buffered by the glibc until it is flushed or the output is set as unbuffered. See setbuf(3). If this option is specified the tasks are executed with a pseudo terminal so that the application output is unbuffered. This option applies to step allocations.

Running the nersc pytorch examples:
cd $SCRATCH
git clone https://github.com/gigantocypris/pytorch-examples.git
cd pytorch-examples
sbatch -N 4 scripts/train_perlmutter.sh configs/mnist.yaml
jobid: 8244723
FAILED

Rolled back to pytorch 1.11.0
sbatch -N 4 scripts/train_perlmutter.sh configs/mnist.yaml
jobid: Submitted batch job 8245647


Learning distributed computing with Pytorch:
Binned references:
	https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
	https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
	https://theaisummer.com/distributed-training-pytorch/
	https://towardsdatascience.com/distribute-your-pytorch-model-in-less-than-20-lines-of-code-61a786e6e7b0

