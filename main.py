import torch
from torch.utils.data import DataLoader
from utils import create_data, NeuralNetwork, get_pde_loss, train, test, get_k0, create_plane_wave_2d, create_plane_wave_3d
import matplotlib.pyplot as plt
import numpy as np
import time

# COMMAND LINE ARGS HERE
# test

# units are microns

# set parameters
batch_size = 400
num_basis = 200 # number of basis functions, N in pde-cl paper
use_pde_cl = True # use the partial differential equation constrained layer
wavelength = 1 # um
n_background = 1.33
use_cpu = False
epochs = 10 # if epochs=0, then load model from model.pth
two_d = True
learning_rate = 1e-3
jitter = 0.015 # jitter for training data

# set the training region
if two_d:
    training_data_x_start = [-14,-14]
    training_data_x_end = [14,14]
    # training_data_x_step = [0.1,0.1]
    training_data_x_step = [0.03,0.03]
else:
    training_data_x_start = [-2,-2,-2]
    training_data_x_end = [2,2,2]
    training_data_x_step = [0.1,0.1,0.1]

# set the test region
if two_d:
    test_data_x_start = [-14,-14]
    test_data_x_end = [14,14]
    test_data_x_step = [0.5,0.5]
else:
    test_data_x_start = [-2,-2,-2]
    test_data_x_end = [2,2,2]
    test_data_x_step = [0.5,0.5,0.5]

# offset if using underdetermined linear system
offset = 0.05

# this ensures that if using Apple Silicon GPU, MPS is activated.
print("Using Apple Silicon GPU? " + str(torch.backends.mps.is_available()))
# this ensures that the current current PyTorch installation was built with MPS activated if using Mac
print("PyTorch install with Apple Silicon GPU support? " + str(torch.backends.mps.is_built()))

dtype = torch.float

# Get cpu, gpu, or mps device for training.
device = torch.device("cuda" 
                      if torch.cuda.is_available() 
                      else "mps"
                      if torch.backends.mps.is_available()
                      else "cpu")

# override
if use_cpu:
    device = torch.device("cpu")

print(f"Using {device} device")

# Training data to compute weights w
# Training data is a list of coordinates
training_data, _ = create_data(training_data_x_start, training_data_x_end, training_data_x_step, device, two_d)

# Training data to compute pde loss
# Training data is a list of coordinates
# This is only used if the linear system is underdetermined
if batch_size<num_basis:
    training_data_2, _ = create_data(np.array(training_data_x_start)+offset, 
                                     np.array(training_data_x_end)+offset, training_data_x_step, device, two_d)

# Test data for validation of pde loss
# Test data is a list of coordinates
test_data, _ = create_data(test_data_x_start, test_data_x_end, test_data_x_step, device, two_d)

# Force num_basis = 1 if not using pde-cl
if not(use_pde_cl):
    num_basis = 1

print(f"Using {num_basis} basis functions")

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
if batch_size<num_basis:
    train_dataloader_2 = DataLoader(training_data_2, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


model = NeuralNetwork(num_basis, two_d)

"""
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)
"""

model.to(device)

                        
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# PDE loss function
def loss_fn(data, u_scatter, data_2): 
    return get_pde_loss(data, 
                        wavelength,
                        n_background,
                        u_scatter,
                        model,
                        device,
                        use_pde_cl,
                        two_d,
                        data_2=data_2,
                        ) 

# Train the PINN
test_loss_vec = []
start = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    if batch_size>num_basis:
        train(train_dataloader, None, model, loss_fn, optimizer, dtype, jitter, device)
    else:
        train(train_dataloader, train_dataloader_2, model, loss_fn, optimizer, dtype, jitter, device)
    test_loss = test(test_dataloader, model, loss_fn, device)
    test_loss_vec.append(test_loss)
    torch.save(model.state_dict(), "model.pth") # save model
    print("Saved PyTorch Model State to model.pth")
print("Done!")
end = time.time()
print("Time to train (s): " + str(start-end))


# Load model
if epochs==0:
    model = NeuralNetwork(num_basis, two_d).to(device)
    model.load_state_dict(torch.load("model.pth"))


# Use loaded model to make predictions
model.eval()

# Visualize the PINN with list of coordinates
eval_data_x_start = training_data_x_start
eval_data_x_end = training_data_x_end
if two_d:
    eval_data_x_step = [0.03,0.03]
else:
    eval_data_x_step = [0.1,0.1,0.1]

eval_data, lengths = create_data(eval_data_x_start, eval_data_x_end, eval_data_x_step, device, two_d)
eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
u_total_all = np.array([])
u_in_all = np.array([])
refractive_index_all = np.array([])
pde_loss = []

with torch.no_grad():
    k0 = get_k0(wavelength)
    total_examples_finished = 0
    size = len(eval_dataloader.dataset)
    for eval_data_i in eval_dataloader:
        if two_d:
            u_in = create_plane_wave_2d(eval_data_i, 
                                        wavelength,
                                        n_background,
                                        device,
                                       )
        else:
            u_in = create_plane_wave_3d(eval_data_i, 
                                        wavelength,
                                        n_background,
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
if two_d:
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
plt.plot(test_loss_vec)
plt.savefig("test_loss.png")
plt.show()

if not(two_d):
    u_total_all = u_total_all[:,:,lengths[2]//2]
    u_in_all = u_in_all[:,:,lengths[2]//2]
    refractive_index_all = refractive_index_all[:,:,lengths[2]//2]

plt.figure()
plt.title('Refractive Index')
sc = plt.imshow(refractive_index_all)
plt.colorbar(sc)
plt.savefig("refractive_index.png")
plt.show()

plt.figure()
plt.title('Magnitude of Total Field')
sc = plt.imshow(np.abs(u_total_all))
plt.colorbar(sc)
plt.savefig("u_total_magnitude.png")
plt.show()

plt.figure()
plt.title('Log Magnitude of Total Field')
sc = plt.imshow(np.log(np.abs(u_total_all)))
plt.colorbar(sc)
plt.savefig("u_total_log_magnitude.png")
plt.show()

plt.figure()
plt.title('Phase of Total Field')
sc = plt.imshow(np.angle(u_total_all))
plt.colorbar(sc)
plt.savefig("u_total_phase.png")
plt.show()

plt.figure()
plt.title('Magnitude Input Wave')
sc = plt.imshow(np.abs(u_in_all))
plt.colorbar(sc)
plt.savefig("u_in_magnitude.png")
plt.show()

plt.figure()
plt.title('Phase Input Wave')
sc = plt.imshow(np.angle(u_in_all))
plt.colorbar(sc)
plt.savefig("u_in_phase.png")
plt.show()


"""
# Scatter Plot results

plt.figure()
plt.title('Magnitude')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=np.abs(u_total_all))
plt.colorbar(sc)
plt.show()

plt.figure()
plt.title('Phase')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=np.angle(u_total_all))
plt.colorbar(sc)
plt.show()

plt.figure()
plt.title('Magnitude Plane Wave')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=np.abs(u_in_all))
plt.colorbar(sc)
plt.show()

plt.figure()
plt.title('Phase Plane Wave')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=np.angle(u_in_all))
plt.colorbar(sc)
plt.show()
"""