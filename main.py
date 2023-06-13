"""
This must be run in the shell/SLURM before running this script:
For NERSC:
export MASTER_ADDR=$(hostname)

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



def get_args():
    ### Command line args ###
    # units are microns

    parser = argparse.ArgumentParser(description='Get command line args')

    parser.add_argument('--bs', type=int, action='store', dest='batch_size',
                        help='batch size', default = 1600)    
    parser.add_argument('--nb', type=int, action='store', dest='num_basis',
                        help='number of basis functions, N in pde-cl paper', default = 200)  
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
                        help='jitter for training data', default = 0.015)
    parser.add_argument('--show', action='store_true', dest='show_figures',
                        help='show figures')
    
    # set the training region
    parser.add_argument('--train_x_start', action='store', dest='training_data_x_start',
                        help='training data x start', nargs='+', default = [-14,-14])
    parser.add_argument('--train_x_end', action='store', dest='training_data_x_end',
                        help='training data x end', nargs='+', default = [14,14])
    parser.add_argument('--train_x_step', action='store', dest='training_data_x_step',
                        help='training data x step', nargs='+', default = [0.03,0.03])
    
    # set the test region
    parser.add_argument('--test_x_start', action='store', dest='test_data_x_start',
                        help='test data x start', nargs='+', default = [-14,-14])
    parser.add_argument('--test_x_end', action='store', dest='test_data_x_end',
                        help='test data x end', nargs='+', default = [14,14])
    parser.add_argument('--test_x_step', action='store', dest='test_data_x_step',
                        help='test data x step', nargs='+', default = [0.5,0.5])   

    # set the evaluation region
    parser.add_argument('--eval_x_start', action='store', dest='eval_data_x_start',
                        help='evaluation data x start', nargs='+', default = [-14,-14])
    parser.add_argument('--eval_x_end', action='store', dest='eval_data_x_end',
                        help='eval data x end', nargs='+', default = [14,14])
    parser.add_argument('--eval_x_step', action='store', dest='eval_data_x_step',
                        help='evaluation data x step', nargs='+', default = [0.03,0.03])  

    parser.add_argument('--load', action='store_true', dest='load_model',
                        help='load model from model.pth')
    parser.add_argument('--dist', action='store_true', dest='use_dist',
                        help='use distributed training')
    parser.add_argument('--checkpoint', action='store', dest='checkpoint_path',
                        help='path to checkpoint', default='model.pth')
    args = parser.parse_args()
    return args


def setup(rank, world_size, fn, args, backend='gloo'):
    os.environ['MASTER_PORT'] = '29500'
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank,world_size, args) # this will be the run function



def cleanup():
    dist.destroy_process_group()

def get_device(args):
    """Get device for non-distributed training"""
    # Get cpu, gpu, or mps device for training.
    device = torch.device("cuda" 
                        if torch.cuda.is_available() 
                        else "cpu")

    print(f"Using {device} device")
    return device

def get_rank(use_dist):
    if use_dist:
        return dist.get_rank()
    else:
        return 0

def partition_dataset(args, world_size):
    """
    Creating and partitioning the dataset
    size is the world size (number of ranks)
    """

    # XXX previous code, delete later
    # train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # if batch_size<num_basis:
    #     train_dataloader_2 = DataLoader(training_data_2, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    


    bsz = args.batch_size//world_size
    partition_sizes = [1.0 / world_size for _ in range(world_size)]

    # Create full dataset

    # Training data to compute weights w
    # Training data is a list of coordinates
    training_data, _ = create_data(args.training_data_x_start, args.training_data_x_end, 
                                   args.training_data_x_step, args.two_d)
    partition = DataPartitioner(training_data, partition_sizes, shuffle=True)
    partition = partition.use(get_rank(args.use_dist))
    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)

    # Training data to compute pde loss
    # Training data is a list of coordinates
    # This is only used if the linear system is underdetermined
    if args.batch_size < args.num_basis:
        training_data_2, _ = create_data(np.array(args.training_data_x_start), 
                                         np.array(args.training_data_x_end), args.training_data_x_step, args.two_d)
        partition = DataPartitioner(training_data_2, partition_sizes, shuffle=True)
        partition = partition.use(get_rank(args.use_dist))
        train_set_2 = torch.utils.data.DataLoader(partition,
                                                  batch_size=bsz,
                                                  shuffle=True)
    else:
        train_set_2 = None

    # Test data for validation of pde loss
    # Test data is a list of coordinates
    test_data, _ = create_data(args.test_data_x_start, args.test_data_x_end, 
                               args.test_data_x_step, args.two_d)
    partition = DataPartitioner(test_data, partition_sizes, shuffle=False)
    partition = partition.use(get_rank(args.use_dist))
    test_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=False)

    return train_set, train_set_2, test_set, bsz



def run(rank, world_size, args,
        dtype = torch.float,
        ):
    if args.use_dist:
        print(f"Running on rank {rank}.")

    train_set, train_set_2, test_set, bsz = partition_dataset(args, world_size)

    # Force num_basis = 1 if not using pde-cl
    if not(args.use_pde_cl):
        args.num_basis = 1

    print(f"Using {args.num_basis} basis functions")

    model = NeuralNetwork(args.num_basis, args.two_d)
    print(model)

    if args.use_dist:
        device = rank #{'cuda:%d' % 0: 'cuda:%d' % rank}
        ddp_model = model
        ddp_model.to(rank)
        #ddp_model = DDP(model, device_ids=[rank])
    else:
        device = get_device(args)
        ddp_model = model
        ddp_model.to(device)
    

    if args.load_model:
        if args.use_dist:  
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} 
            ddp_model.load_state_dict(
                torch.load(args.checkpoint_path, map_location=device)
            )
        else: 
            model.load_state_dict(torch.load(args.checkpoint_path))

    
    # PDE loss function
    def loss_fn(data, u_scatter, data_2): 
        return get_pde_loss(data, 
                            args.wavelength,
                            args.n_background,
                            u_scatter,
                            ddp_model,
                            device,
                            args.use_pde_cl,
                            args.two_d,
                            data_2=data_2,
                            ) 
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    # Train the PINN
    test_loss_vec = []
    start = time.time()
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_set, train_set_2, model, loss_fn, optimizer, dtype, args.jitter, device, args.use_dist)
        test_loss = test(test_set, model, loss_fn, device)
        test_loss_vec.append(test_loss)
        # Automatically synced here, don't need barrier
        if rank == 0:
            torch.save(ddp_model.state_dict(), args.checkpoint_path) # save model
        print("Saved PyTorch Model State to: " + args.checkpoint_path)
    torch.save(test_loss_vec, "test_loss_vec_" + str(rank) + ".pth") # save test loss
    print("Done! Rank: " + str(rank))

    if args.use_dist:
        cleanup()

def visualize(args):
    """
    Visualize the PINN with list of evaluation coordinates
    Not yet implemented with distributed computing
    """
    device = get_device(args)
    eval_data, lengths = create_data(args.eval_data_x_start, args.eval_data_x_end, 
                                     args.eval_data_x_step, args.two_d)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = NeuralNetwork(args.num_basis, args.two_d).to(device)
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

    # Use loaded model to make predictions
    model.eval()

    # Visualize the PINN with list of coordinates

    u_total_all = np.array([])
    u_in_all = np.array([])
    refractive_index_all = np.array([])
    pde_loss = []

    with torch.no_grad():
        k0 = get_k0(args.wavelength)
        total_examples_finished = 0
        size = len(eval_dataloader.dataset)
        for eval_data_i in eval_dataloader:
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
            
            pde_loss_i, u_total, u_scatter, refractive_index = loss_fn(eval_data_i, 
                                                                    u_scatter_test,
                                                                    data_2=None,
                                                                    )
            pde_loss.append(pde_loss_i.cpu().numpy())

            u_total_all = np.concatenate((u_total_all,u_total.cpu().numpy()), axis=0)
            u_in_all = np.concatenate((u_in_all, u_in.cpu().numpy()), axis=0)
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
        refractive_index_all = np.reshape(refractive_index_all, [lengths[0],lengths[1]])
    else:  
        eval_data = np.reshape(eval_data, [lengths[0],lengths[1],lengths[2],3]) # use as a check
        u_total_all = np.reshape(u_total_all, [lengths[0],lengths[1],lengths[2]])
        u_in_all = np.reshape(u_in_all, [lengths[0],lengths[1],lengths[2]])
        refractive_index_all = np.reshape(refractive_index_all, [lengths[0],lengths[1],lengths[2]])

    np.save("u_total_all.npy", u_total_all)
    np.save("u_in_all.npy", u_in_all)


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
    if (torch.cuda.device_count() > 1) and args.use_dist:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
        args.use_dist = False

    start = time.time()
    if args.use_dist:
        processes = []
        mp.set_start_method("spawn")
        for rank in range(world_size):
            p = mp.Process(target=setup, args=(rank, world_size, run, args))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()


        # mp.spawn(run,
        #         args=(world_size,args,torch.float), # arguments passed to demo_fn after the rank argument
        #         nprocs=world_size, # number of processes to spawn
        #         join=True)
    else:
        rank = 0
        run(rank, world_size, args,
            )
        
    visualize(args)
    end = time.time()
    print("Time to train (s): " + str(end-start))