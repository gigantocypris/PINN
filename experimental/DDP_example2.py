"""
This must be run in the shell/SLURM script:
export MASTER_ADDR=$(hostname)

Reference: https://pytorch.org/tutorials/intermediate/dist_tuto.html
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_blocking_p_p(rank_size):
    """Blocking point-to-point communication"""
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receives the tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def run(rank,size):
    """Distributed function to be implemented later"""
    pass

def init_process(rank, size, fn, backend='gloo'):
    """Initialize the distributed environment"""
    os.environ['MASTER_PORT'] = 29500
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size) # this will be the run function

print("hello")

if __name__ == "main":
    print("hello")
    size = 2 # total number of ranks
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_blocking_p_p))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()