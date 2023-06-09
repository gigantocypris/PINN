"""
Hello, World DDP tutorial
https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

This must be run in the shell/SLURM before running this script:
export MASTER_ADDR=$(hostname)
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

CHECKPOINT_PATH = "model.checkpoint" 

def setup(rank, world_size):
    os.environ['MASTER_PORT'] = '29500'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10,10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10,5)
    
    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20,10))
    labels = torch.randn(20,5).to(rank)
    loss = loss_fn(outputs,labels)
    # print(loss)
    loss.backward()
    optimizer.step()


    cleanup()



def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier to make sure that process n>0 load the model after process 0 saves it
    dist.barrier()

    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20,10))
    labels = torch.randn(20,5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()
    ''' 
    # Automatically synced here, don't need barrier
    if rank == 0:
        os.remove(CHECKPOINT_PATH)
    '''
    cleanup()



def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,), # arguments passed to demo_fn after the rank argument
             nprocs=world_size, # number of processes to spawn
             join=True)


if __name__ == "__main__":
    world_size = 2
    # fn = demo_basic
    fn = demo_checkpoint
    run_demo(fn, world_size)