import torch
import numpy as np
from random import Random
from .distributed import get_rank

def create_data(min_vals, max_vals, spacings, two_d):
    """Create a list of coordinates from a grid"""
    data = []
    lengths = []
    for i in range(len(spacings)):
        data_i = torch.arange(min_vals[i],max_vals[i],spacings[i])
        data.append(data_i)
        lengths.append(len(data_i))
    if two_d:
        data_xm, data_ym = torch.meshgrid(data[0], data[1], indexing='ij')
        data = torch.stack((data_xm, data_ym), dim=2)
        data = torch.reshape(data, (-1,2))
    else:
        data_xm, data_ym, data_zm = torch.meshgrid(data[0], data[1], data[2], indexing='ij')
        data = torch.stack((data_xm, data_ym, data_zm), dim=3)
        data = torch.reshape(data, (-1,3))
    return data, lengths


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
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, shuffle=True):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        if shuffle:
            rng.shuffle(indexes)
        self.indexes = indexes

        for ind,frac in enumerate(sizes):
            part_len = int(frac*data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
        

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
    def use_all(self):
        return Partition(self.data, self.indexes)


def partition_dataset(args, world_size):
    """
    Creating and partitioning the dataset
    size is the world size (number of ranks)
    """

    partition_sizes = [1.0 / world_size for _ in range(world_size)]

    # Create full dataset

    # Training data to compute weights w
    # Training data is a list of coordinates
    training_data, _ = create_data(args.data_x_start, args.data_x_end, 
                                   args.training_data_x_step, args.two_d)

    training_partition = DataPartitioner(training_data, partition_sizes, shuffle=True)


    # Training data to compute pde loss
    # Training data is a list of coordinates
    # This is only used if the linear system is underdetermined
    if args.batch_size < args.num_basis and args.use_pde_cl:
        training_data_2, _ = create_data(np.array(args.data_x_start), 
                                         np.array(args.data_x_end), args.training_data_x_step, args.two_d)
        training_2_partition = DataPartitioner(training_data_2, partition_sizes, shuffle=True)
    else:
        training_2_partition = None

    # Test data for validation of pde loss
    # Test data is a list of coordinates
    test_data, _ = create_data(args.data_x_start, args.data_x_end, 
                               args.test_data_x_step, args.two_d)
    test_partition = DataPartitioner(test_data, partition_sizes, shuffle=True)


    return training_partition, training_2_partition, test_partition

def get_train_test_sets(args, training_partition, training_2_partition, test_partition):
    if args.use_pde_cl:
        training_partition = training_partition.use_all()
    else:
        training_partition = training_partition.use(get_rank()[0])
    train_set = torch.utils.data.DataLoader(training_partition,
                                            batch_size=args.batch_size,
                                            shuffle=True)
    if args.batch_size < args.num_basis and args.use_pde_cl:
        training_2_partition = training_2_partition.use_all()
        train_set_2 = torch.utils.data.DataLoader(training_2_partition,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
    else:
        train_set_2 = None
    

    if args.use_pde_cl:
        test_partition = test_partition.use_all()
    else:
        test_partition = test_partition.use(get_rank()[0])
    test_set = torch.utils.data.DataLoader(test_partition,
                                            batch_size=args.batch_size,
                                            shuffle=True)
    return train_set, train_set_2, test_set