import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_img(mat, descriptor):
    plt.figure()
    plt.title(descriptor)
    sc = plt.imshow(mat)
    plt.colorbar(sc)
    plt.savefig(descriptor + ".png")

def plot_all(args, world_size, lengths, u_total_all, u_scatter_all, u_in_all, refractive_index_all):
    # Plot results
    plt.figure()
    plt.title('Test Loss')
    for i in range(world_size):
        test_loss_vec = torch.load("test_loss_vec_" + str(i) + ".pth")
        plt.plot(test_loss_vec)
    plt.savefig("test_loss.png")

    if not(args.two_d):
        u_total_all = u_total_all[:,:,lengths[2]//2]
        u_in_all = u_in_all[:,:,lengths[2]//2]
        refractive_index_all = refractive_index_all[:,:,lengths[2]//2]

    plot_img(refractive_index_all, "refractive_index")

    plot_img(np.abs(u_total_all), "u_total_magnitude")
    plot_img(np.angle(u_total_all), "u_total_phase")

    plot_img(np.abs(u_scatter_all), "u_scatter_magnitude")
    plot_img(np.angle(u_scatter_all), "u_scatter_phase")

    plot_img(np.abs(u_in_all), "u_in_magnitude")
    plot_img(np.angle(u_in_all), "u_in_phase")
