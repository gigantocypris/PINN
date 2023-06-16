"""
To run on NERSC:
export MASTER_ADDR=$(hostname)
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def get_nodelist():
    slurm_job_nodelist = os.environ.get('SLURM_JOB_NODELIST')
    nodes = []
    prefix='nid'
    if slurm_job_nodelist:
        # Remove any enclosing brackets and split into individual nodes
        slurm_job_nodelist = slurm_job_nodelist.strip('nid').strip('[]').split(',')

        for node_spec in slurm_job_nodelist:
            if '-' in node_spec:
                # Expand node ranges, e.g., "001-003" becomes ["nid001", "nid002", "nid003"]
                node_range = node_spec.split('-')
                start = int(node_range[0])
                end = int(node_range[1])
                nodes.extend([prefix+str(i) for i in range(start, end + 1)])
            else:
                nodes.append(prefix+str(node_spec.zfill(6)))

    print(nodes)
    return nodes

def init_process(rank, world_size, fn, head_node, backend='nccl'):
    os.environ['MASTER_ADDR'] = head_node
    os.environ['MASTER_PORT'] = '29510'
    dist.init_process_group(backend=backend, 
                            rank=int(os.environ['SLURM_PROCID']), 
                            world_size=world_size)
    fn(rank,world_size) # this will be the run function

def run(rank,world_size):
    # Set the GPU device for this rank
    device = torch.device(f'cuda:{rank}')
    x = torch.Tensor([1]).to(device)
    rank_confirm = dist.get_rank()
    print(f"Hello from process {rank}! Confirming rank {rank_confirm}. Running on GPU: {device}. Tensor {x}")

def main():
    # Get the total number of processes
    world_size = 4 #int(os.environ['SLURM_NTASKS'])

    """Initialize the distributed environment"""
    node_list = get_nodelist()


    # Spawn the processes
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process,
                       args=(rank, world_size, run, node_list[0]))
        p.start()
        processes.append(p)
        

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
