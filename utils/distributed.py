import os
import torch
import torch.distributed as dist

def setup(rank, world_size, fn, args, 
          training_partition, training_2_partition, test_partition,
          backend='nccl'):
    os.environ['MASTER_PORT'] = '29500'

    # Get the SLURM_PROCID for the current process
    # proc_id = int(os.environ['SLURM_PROCID'])

    # print("Hello from " + str(proc_id))
    # print(get_rank())

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank,world_size, args,
       training_partition, training_2_partition, test_partition) # this will be the run function

def get_rank():
    return dist.get_rank()

def cleanup():
    dist.destroy_process_group()

def get_device(args, force_cpu=False):
    """Get device for non-distributed training"""
    # Get cpu, gpu, or mps device for training.
    device = torch.device("cuda" 
                        if torch.cuda.is_available() 
                        else "cpu")
    if force_cpu:
        device = torch.device("cpu")

    print("Using " + str(device) + " device")
    return device

def average_gradients(model):
    """Gradient averaging."""
    world_size = dist.get_world_size()
    for param in model.parameters():
        if type(param) is torch.Tensor:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            param.grad.data /= world_size