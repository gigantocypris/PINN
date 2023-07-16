"""
This must be run in the shell/SLURM before running this script:
For NERSC:
export MASTER_ADDR=$(hostname)
export SLURM_NTASKS=4

For other servers:
export MASTER_ADDR=localhost
"""
import os
import numpy as np
import time
import argparse
import torch
# from torch.autograd import Variable

from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from utils.physics import get_pde_loss, get_k0, create_plane_wave_2d, create_plane_wave_3d
from utils.visualize import plot_all
from utils.dataset import get_train_test_sets, partition_dataset, create_data
from utils.distributed import setup, get_rank, cleanup, average_gradients, get_device
from models import Siren, NeuralNetwork


def get_args():
    ### Command line args ###
    # units are microns

    parser = argparse.ArgumentParser(description='Get command line args')

    parser.add_argument('--bs', type=int, action='store', dest='batch_size',
                        help='batch size (per gpu with pde-cl, total without)', default = 4)  
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
                        help='jitter for training data', default = 0.2)
    
    # set the region
    parser.add_argument('--x_start',type=float, action='store', dest='data_x_start',
                        help='boundary x start', nargs='+', default = [-10.0,-10.0])
    parser.add_argument('--x_end', type=float, action='store', dest='data_x_end',
                        help='boundary data x end', nargs='+', default = [10.0,10.0])
    
    # set the pml thickness
    parser.add_argument('--pml_thickness', type=float, action='store', dest='pml_thickness',
                        help='pml thickness', nargs='+', default = [2.0,2.0])
    
    # set the training spacing
    parser.add_argument('--train_x_step', type=float, action='store', dest='training_data_x_step',
                        help='training data x step', nargs='+', default = [0.2,0.2])
    
    # set the test spacing
    parser.add_argument('--test_x_step', type=float, action='store', dest='test_data_x_step',
                        help='test data x step', nargs='+', default = [0.2,0.2])   

    # set the evaluation region subset spacing for evaluting w
    parser.add_argument('--eval_x_step_subset', type=float, action='store', dest='eval_data_x_step_subset',
                        help='evaluation data x step', nargs='+', default = [0.2,0.2])  

    # set the evaluation region spacing for final visualization
    parser.add_argument('--eval_x_step', type=float, action='store', dest='eval_data_x_step',
                        help='evaluation data x step', nargs='+', default = [0.01,0.01])  

    parser.add_argument('--load', action='store_true', dest='load_model',
                        help='load model from model.pth')
    
    parser.add_argument('--checkpoint', action='store', dest='checkpoint_path',
                        help='path to checkpoint', default='model.pth')
    args = parser.parse_args()
    return args

def run(rank, world_size, args,
        training_partition, training_2_partition, test_partition,
        dtype = torch.float,
        ):
    
    print("Running on rank " + str(rank) + ". Running on rank " + str(get_rank()))

    train_set, train_set_2, test_set = get_train_test_sets(args, training_partition, training_2_partition, test_partition)
    
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
    model.to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.checkpoint_path))

    # PDE loss function
    def loss_fn(data, u_scatter, data_2): 
        return get_pde_loss(args,
                            data, 
                            u_scatter,
                            model,
                            device,
                            data_2=data_2,
                            ) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    # Train the PINN
    test_loss_vec = []
    start = time.time()
    for t in range(args.epochs):
        if rank == 0:
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

def train(dataloader, 
          dataloader_2,
          model, 
          loss_fn, 
          optimizer,
          dtype,
          jitter,
          device,
          ):
    
    """Train the model for one epoch"""
    if dataloader_2 is not None:
        dataloader_2_iter = iter(dataloader_2)
    size = len(dataloader.dataset)
    model.train()
    total_examples_finished = 0
    for data in dataloader:
        # data = Variable(data)
        data = data.to(device)
        rand_1 = jitter*(2*torch.rand(data.shape, dtype=dtype, device=device) - 1.0)
        # rand_1 = jitter*torch.randn(data.shape, dtype=dtype, device=device)
        if dataloader_2 is not None:
            data_2 = next(dataloader_2_iter)
            # data_2 = Variable(data_2)
            rand_2 = jitter*(2*torch.rand(data_2.shape, dtype=dtype, device=device) - 1.0)
            # rand_2 = jitter*torch.randn(data_2.shape, dtype=dtype, device=device)
            data_2 = data_2.to(device)
            data_2 += rand_2
        else:
            data_2 = None
        
        
        data += rand_1
        # Compute prediction error
        u_scatter = model(data)
        pde_loss, _, _, _, _ = loss_fn(data, 
                                    u_scatter,
                                    data_2.to(device) if data_2 is not None else None,
                                   )
        pde_loss = pde_loss/len(data)
        # Backpropagation
        optimizer.zero_grad()
        pde_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        average_gradients(model)
        torch.distributed.barrier()
        optimizer.step()
        total_examples_finished += len(data)
        print(f"{device}: loss: {pde_loss.item():>7f}  [{total_examples_finished:>5d}/{size:>5d}]")

def test(dataloader, 
         model, 
         loss_fn, 
         device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            
            u_scatter = model(data)
            pde_loss, _, _, _, _ = loss_fn(data, 
                                     u_scatter,
                                     data_2=None,
                                    )
            test_loss += pde_loss.item()
            
    test_loss /= size
    print(f"Avg test loss: {test_loss:>8f}")
    return test_loss

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
    eval_data_subset, _ = create_data(args.data_x_start, args.data_x_end, 
                                    args.eval_data_x_step_subset, args.two_d)
    
    eval_data, lengths = create_data(args.data_x_start, args.data_x_end, 
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
        return get_pde_loss(args,
                            data, 
                            u_scatter,
                            model,
                            device,
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
    plot_all(args, world_size, lengths, u_total_all, u_scatter_all, u_in_all, refractive_index_all)


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
        
    visualize(args)
    end = time.time()
    print("Time to train (s): " + str(end-start))