"""
This must be run in the shell/SLURM script:
export MASTER_ADDR=$(hostname)

Listing available CUDA devices:
> python
>>> import torch
>>> num_of_gpus = torch.cuda.device_count()
>>> print(num_of_gpus)

Use an interactive session to run:
salloc -N 1 --time=60 -C gpu -A m3562_g --qos=interactive
 --ntasks-per-gpu=2

Reference: https://pytorch.org/tutorials/intermediate/dist_tuto.html
https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
import torch.multiprocessing as mp
from torchvision import datasets, transforms

import time

def run_blocking_p_p(rank, size):
    """Blocking point-to-point communication"""
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receives the tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ' + str(rank) + ' has data ' + str(tensor[0]))

def run_nonblocking_p_p(rank, size):
    """Non-blocking point-to-point communication"""
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ' + str(rank) + ' has data ' + str(tensor[0]))

def run_all_reduce(rank, size):
    """All-reduce example, sum of all tensors on all processes"""
    group = dist.new_group([0,1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ' + str(rank) + ' has data ' + str(tensor[0]))

def run(rank,size):
    """Distributed function to be implemented later"""
    pass

def init_process(rank, size, fn, backend='gloo'):
    """Initialize the distributed environment"""
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size) # this will be the run function

class Partition(object):
    """Dataset partitioning helper"""
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self,index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """Partitions a dataset into different chunks"""
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac*data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class Net(nn.Module):
    """Network architecture."""
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def partition_dataset():
    """Partitioning MNIST"""
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,),(0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128//size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)
    return train_set, bsz

def average_gradients(model):
    """Gradient averaging."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        if type(param) is torch.Tensor:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            param.grad.data /= size

def run_mnist(rank, size):
    """Distributed Synchronous SGD Example"""
    print('Hello from rank: ' + str(rank))
    device = torch.device("cuda:{}".format(rank))
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    
    num_batches = ceil(len(train_set.dataset)/float(bsz))

    for epoch in range(10):
        epoch_loss = 0.0
        print('rank is ' + str(rank) +' and epoch is ' + str(epoch))
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)


if __name__ == "__main__":
    print("hello")
    start_time = time.time()
    size = 4 # total number of ranks
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_mnist))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    end_time = time.time()
    print('Total training time was: ' + str(end_time-start_time))