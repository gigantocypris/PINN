"""
This must be run in the shell/SLURM before running this script:
For NERSC:
export MASTER_ADDR=$(hostname)
For interactive session:
export SLURM_NTASKS=4

For other servers:
export MASTER_ADDR=localhost

for default operation without distribution:
python main.py --upc --2d

for distributed operation:
python main.py --upc --2d --dist
"""
import os
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import create_data, NeuralNetwork, get_pde_loss, train, test, get_k0, create_plane_wave_2d, create_plane_wave_3d, DataPartitioner
from siren import Siren


def get_args():
    ### Command line args ###
    # units are microns

    parser = argparse.ArgumentParser(description='Get command line args')

    parser.add_argument('--bs', type=int, action='store', dest='batch_size',
                        help='batch size per gpu', default = 8836)  
    parser.add_argument('--nb', type=int, action='store', dest='num_basis',
                        help='number of basis functions, N in pde-cl paper', default = 200)  
    parser.add_argument('--siren', action='store_true', dest='use_siren',
                        help='use the siren architecture')
    parser.add_argument('--upc', action='store_true', dest='use_pde_cl', 
                        help='use the partial differential equation constrained layer') 
    parser.add_argument('-w', type=float, action='store', dest='wavelength', 
                        help='wavelength of light in free space', default=1)
    parser.add_argument('--nback', type=float, action='store', dest='n_background', 
                        help='refractive index of the background medium', default=1.33)
    parser.add_argument('--epochs', type=int, action='store', dest='epochs',
                        help='number of epochs', default = 1)
    parser.add_argument('--2d', action='store_true', dest='two_d',
                        help='simulation in 2d as opposed to 3d')
    parser.add_argument('--lr', type=float, action='store', dest='learning_rate',
                        help='learning rate', default = 1e-3)
    parser.add_argument('-j', type=float, action='store', dest='jitter',
                        help='jitter for training data', default = 0.5)
    parser.add_argument('--show', action='store_true', dest='show_figures',
                        help='show figures')
    
    # set the region
    parser.add_argument('--x_start', action='store', dest='data_x_start',
                        help='boundary x start', nargs='+', default = [-14.0,-14.0])
    parser.add_argument('--x_end', action='store', dest='data_x_end',
                        help='boundary data x end', nargs='+', default = [14.0,14.0])
    
    # set the training spacing
    parser.add_argument('--train_x_step', action='store', dest='training_data_x_step',
                        help='training data x step', nargs='+', default = [0.015,0.015])
    
    # set the test spacing
    parser.add_argument('--test_x_step', action='store', dest='test_data_x_step',
                        help='test data x step', nargs='+', default = [0.3,0.3])   

    # set the evaluation region subset spacing for evaluting w
    parser.add_argument('--eval_x_step_subset', action='store', dest='eval_data_x_step_subset',
                        help='evaluation data x step', nargs='+', default = [0.15,0.15])  

    # set the evaluation region spacing for final visualization
    parser.add_argument('--eval_x_step', action='store', dest='eval_data_x_step',
                        help='evaluation data x step', nargs='+', default = [0.05,0.05])  

    parser.add_argument('--load', action='store_true', dest='load_model',
                        help='load model from model.pth')
    
    parser.add_argument('--checkpoint', action='store', dest='checkpoint_path',
                        help='path to checkpoint', default='model.pth')
    args = parser.parse_args()
    return args


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

def get_rank():
    return dist.get_rank()


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
    test_data, _ = create_data(args.test_data_x_start, args.test_data_x_end, 
                               args.test_data_x_step, args.two_d)
    test_partition = DataPartitioner(test_data, partition_sizes, shuffle=True)


    return training_partition, training_2_partition, test_partition



def run(rank, world_size, args,
        training_partition, training_2_partition, test_partition,
        dtype = torch.float,
        ):
    

    print("Running on rank " + str(rank) + ". Running on rank " + str(get_rank()))

    if args.use_pde_cl:
        training_partition = training_partition.use_all()
    else:
        training_partition = training_partition.use(get_rank())
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
        test_partition = test_partition.use(get_rank())
    test_set = torch.utils.data.DataLoader(test_partition,
                                            batch_size=args.batch_size,
                                            shuffle=True)
    
    # Force num_basis = 1 if not using pde-cl
    if not(args.use_pde_cl):
        args.num_basis = 1

    print("Using " + str(args.num_basis) + " basis functions")

    if args.use_siren:
        model = Siren(args.num_basis, args.two_d)
    else:
        model = NeuralNetwork(args.num_basis, args.two_d)
    print(model)


    device = torch.device(f'cuda:{rank}')
    # device = get_device(args)
    model.to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.checkpoint_path))

    # PDE loss function
    def loss_fn(data, u_scatter, data_2): 
        return get_pde_loss(data, 
                            args.wavelength,
                            args.n_background,
                            u_scatter,
                            model,
                            device,
                            args.use_pde_cl,
                            args.two_d,
                            data_2=data_2,
                            ) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    # Train the PINN
    test_loss_vec = []
    start = time.time()
    for t in range(args.epochs):
        print("Epoch " + str(t+1) + "\n-------------------------------")
        train(train_set, train_set_2, model, loss_fn, optimizer, dtype, args.jitter, device)
        test_loss = test(test_set, model, loss_fn, device)
        test_loss_vec.append(test_loss)
        # Automatically synced here, don't need barrier
        if rank == 0:
            torch.save(model.state_dict(), args.checkpoint_path) # save model
        print("Saved PyTorch Model State to: " + args.checkpoint_path)
    torch.save(test_loss_vec, "test_loss_vec_" + str(rank) + ".pth") # save test loss
    print("Done! Rank: " + str(rank))

    cleanup()

def evaluate(eval_data_i,
             device,
             model,
             loss_fn,
             w,
             args,
             ):
    eval_data_i = eval_data_i.to(device)

    if args.two_d:
        u_in = create_plane_wave_2d(eval_data_i, 
                                    args.wavelength,
                                    args.n_background,
                                    device,
                                )
    else:
        u_in = create_plane_wave_3d(eval_data_i, 
                                    args.wavelength,
                                    args.n_background,
                                    device,
                                )
    
    u_scatter_test = model(eval_data_i)
    
    pde_loss_i, u_total, u_scatter, refractive_index, w = loss_fn(eval_data_i, 
                                                                    u_scatter_test,
                                                                    data_2=None,
                                                                    w=w,
                                                                    )
    return pde_loss_i, u_total, u_scatter, refractive_index, w, u_in

def visualize(args,
              ):
    """
    Visualize the PINN with list of evaluation coordinates
    Not yet implemented with distributed computing
    """
    device = get_device(args)
    # Solve the linear system for a subset of the points, use those weights for all points
    eval_data_subset, _ = create_data(args.eval_data_x_start_subset, args.eval_data_x_end_subset, 
                                    args.eval_data_x_step_subset, args.two_d)
    
    eval_data, lengths = create_data(args.eval_data_x_start, args.eval_data_x_end, 
                                     args.eval_data_x_step, args.two_d)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    
    # Load model
    
    # Force num_basis = 1 if not using pde-cl
    if not(args.use_pde_cl):
        args.num_basis = 1

    if args.use_siren:
        model = Siren(args.num_basis, args.two_d)
    else:
        model = NeuralNetwork(args.num_basis, args.two_d)

    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))

    # PDE loss function
    def loss_fn(data, u_scatter, data_2,w=None): 
        return get_pde_loss(data, 
                            args.wavelength,
                            args.n_background,
                            u_scatter,
                            model,
                            device,
                            args.use_pde_cl,
                            args.two_d,
                            data_2=data_2,
                            w=w,
                            ) 

    # Use loaded model to make predictions
    model.eval()

    # Visualize the PINN with list of coordinates

    u_total_all = np.array([])
    u_in_all = np.array([])
    u_scatter_all = np.array([])
    refractive_index_all = np.array([])
    pde_loss = []
    k0 = get_k0(args.wavelength)

    with torch.no_grad():
        
        pde_loss_i, u_total, u_scatter, refractive_index, w, u_in = evaluate(eval_data_subset,
                                                                             device,
                                                                             model,
                                                                             loss_fn,
                                                                             None,
                                                                             args,
                                                                            )

        total_examples_finished = 0
        size = len(eval_dataloader.dataset)
        for eval_data_i in eval_dataloader:
            pde_loss_i, u_total, u_scatter, refractive_index, w, u_in = evaluate(eval_data_i,
                                                                           device,
                                                                           model,
                                                                           loss_fn,
                                                                           w,
                                                                           args,
                                                                          )

            pde_loss.append(pde_loss_i.cpu().numpy())

            u_total_all = np.concatenate((u_total_all,u_total.cpu().numpy()), axis=0)
            u_in_all = np.concatenate((u_in_all, u_in.cpu().numpy()), axis=0)
            u_scatter_all = np.concatenate((u_scatter_all, u_scatter.cpu().numpy()), axis=0)
            refractive_index_all = np.concatenate((refractive_index_all, refractive_index.cpu().numpy()), axis=0)
            total_examples_finished += len(eval_data_i)
            print(f"loss: {pde_loss_i/len(eval_data_i):>7f}  [{total_examples_finished:>5d}/{size:>5d}]")

    print(f"Final eval pde loss is {np.sum(pde_loss)/len(eval_data)}")

    eval_data = eval_data.cpu().numpy()


    # reshape the output matrices
    if args.two_d:
        eval_data = np.reshape(eval_data, [lengths[0],lengths[1],2]) # use as a check
        u_total_all = np.reshape(u_total_all, [lengths[0],lengths[1]])
        u_in_all = np.reshape(u_in_all, [lengths[0],lengths[1]])
        u_scatter_all = np.reshape(u_scatter_all, [lengths[0],lengths[1]])
        refractive_index_all = np.reshape(refractive_index_all, [lengths[0],lengths[1]])
    else:  
        eval_data = np.reshape(eval_data, [lengths[0],lengths[1],lengths[2],3]) # use as a check
        u_total_all = np.reshape(u_total_all, [lengths[0],lengths[1],lengths[2]])
        u_in_all = np.reshape(u_in_all, [lengths[0],lengths[1],lengths[2]])
        u_scatter_all = np.reshape(u_scatter_all, [lengths[0],lengths[1],lengths[2]])
        refractive_index_all = np.reshape(refractive_index_all, [lengths[0],lengths[1],lengths[2]])

    np.save("u_total_all.npy", u_total_all)
    np.save("u_in_all.npy", u_in_all)
    np.save("u_scatter_all.npy", u_scatter_all)


    # Plot results
    plt.figure()
    plt.title('Test Loss')
    for i in range(world_size):
        test_loss_vec = torch.load("test_loss_vec_" + str(i) + ".pth")
        plt.plot(test_loss_vec)
    plt.savefig("test_loss.png")
    if args.show_figures:
        plt.show()

    if not(args.two_d):
        u_total_all = u_total_all[:,:,lengths[2]//2]
        u_in_all = u_in_all[:,:,lengths[2]//2]
        refractive_index_all = refractive_index_all[:,:,lengths[2]//2]

    plt.figure()
    plt.title('Refractive Index')
    sc = plt.imshow(refractive_index_all)
    plt.colorbar(sc)
    plt.savefig("refractive_index.png")
    if args.show_figures:
        plt.show()

    plt.figure()
    plt.title('Magnitude of Total Field')
    sc = plt.imshow(np.abs(u_total_all))
    plt.colorbar(sc)
    plt.savefig("u_total_magnitude.png")
    if args.show_figures:
        plt.show()

    plt.figure()
    plt.title('Log Magnitude of Total Field')
    sc = plt.imshow(np.log(np.abs(u_total_all)))
    plt.colorbar(sc)
    plt.savefig("u_total_log_magnitude.png")
    if args.show_figures:
        plt.show()

    plt.figure()
    plt.title('Phase of Total Field')
    sc = plt.imshow(np.angle(u_total_all))
    plt.colorbar(sc)
    plt.savefig("u_total_phase.png")
    if args.show_figures:
        plt.show()

    plt.figure()
    plt.title('Magnitude of Scattered Field')
    sc = plt.imshow(np.abs(u_scatter_all))
    plt.colorbar(sc)
    plt.savefig("u_scatter_magnitude.png")
    if args.show_figures:
        plt.show()

    plt.figure()
    plt.title('Log Magnitude of Scattered Field')
    sc = plt.imshow(np.log(np.abs(u_scatter_all)))
    plt.colorbar(sc)
    plt.savefig("u_scatter_log_magnitude.png")
    if args.show_figures:
        plt.show()

    plt.figure()
    plt.title('Phase of Scattered Field')
    sc = plt.imshow(np.angle(u_scatter_all))
    plt.colorbar(sc)
    plt.savefig("u_scatter_phase.png")
    if args.show_figures:
        plt.show()

    plt.figure()
    plt.title('Magnitude Input Wave')
    sc = plt.imshow(np.abs(u_in_all))
    plt.colorbar(sc)
    plt.savefig("u_in_magnitude.png")
    if args.show_figures:
        plt.show()

    plt.figure()
    plt.title('Phase Input Wave')
    sc = plt.imshow(np.angle(u_in_all))
    plt.colorbar(sc)
    plt.savefig("u_in_phase.png")
    if args.show_figures:
        plt.show()




if __name__=='__main__':

    args = get_args()

    print(str(torch.cuda.device_count()) + " GPUs detected!")

    # world_size = torch.cuda.device_count()
    world_size = int(os.environ['SLURM_NTASKS'])

    print('world_size is: ' + str(world_size))

    training_partition, training_2_partition, test_partition = partition_dataset(args, world_size)
    start = time.time()

    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=setup, args=(rank, world_size, run, args,
                                            training_partition, training_2_partition, test_partition,
                                            ))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


    # rank = 0
    # run(rank, world_size, args,
    #     training_partition, training_2_partition, test_partition,
    #     )
        
    visualize(args)
    end = time.time()
    print("Time to train (s): " + str(end-start))