import torch
from torch.utils.data import DataLoader
from pinn_utils import create_data, NeuralNetwork, get_pde_loss, train, test, get_k0, create_plane_wave
import matplotlib.pyplot as plt
import numpy as np

"""
Tests to write:
Scattered field is zero when there is no scatterer
Scattered field is nonzero when there is a scatterer and the incident field is nonzero
Scattered field is zero when there is a scatterer and the incident field is zero
Scattered field matches analytic solution for simple geometries
Distributed code matches results of non-distributed code
"""

# units are microns

batch_size = 100
num_basis = 200 # number of basis functions, N in pde-cl paper
use_pde_cl = True
wavelength = 2.1 # um
n_background = 1
use_cpu = True
epochs = 2

training_data_x_start = [-2,-2,0]
training_data_x_end = [2,2,0.5]
training_data_x_step = [0.1,0.1,0.5]

offset_2 = 0.05

test_data_x_start = [-2,-2,0]
test_data_x_end = [2,2,0.5]
test_data_x_step = [0.5,0.5,0.5]

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

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
training_data = create_data(training_data_x_start, training_data_x_end, training_data_x_step, device)


# Training data to compute pde loss
if batch_size<num_basis:
    training_data_2 = create_data(np.array(training_data_x_start)+offset_2, 
                                np.array(training_data_x_end)+offset_2, training_data_x_step, device)

test_data = create_data(test_data_x_start, test_data_x_end, test_data_x_step, device)

# Force num_basis = 1 if not using pde-cl
if not(use_pde_cl):
    num_basis = 1

print(f"Using {num_basis} basis functions")

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
if batch_size<num_basis:
    train_dataloader_2 = DataLoader(training_data_2, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


model = NeuralNetwork(num_basis).to(device)
print(model)


optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def loss_fn(data, u_scatter, data_2): 
    return get_pde_loss(data, 
                        wavelength,
                        n_background,
                        u_scatter,
                        model,
                        device,
                        use_pde_cl,
                        data_2=data_2,
                        ) 

test_loss_vec = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    if batch_size>num_basis:
        train(train_dataloader, None, model, loss_fn, optimizer, device)
    else:
        train(train_dataloader, train_dataloader_2, model, loss_fn, optimizer, device)
    test_loss = test(test_dataloader, model, loss_fn, device)
    test_loss_vec.append(test_loss)
print("Done!")

# Save model

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# Load model

model = NeuralNetwork(num_basis)
model.load_state_dict(torch.load("model.pth"))


# Use loaded model to make predictions

model.eval()

# eval_data = create_data(training_data_x_start, training_data_x_end, training_data_x_step, device)
eval_data = create_data([-2,-2,0], [2,2,0.5], [0.1,0.1,0.5], device)

with torch.no_grad():
    k0 = get_k0(wavelength)
    u_in = create_plane_wave(eval_data, 
                             wavelength,
                             n_background,
                             device,
                            )
    
    u_scatter_test = model(eval_data)
    
    pde_loss, u_total, u_scatter = loss_fn(eval_data, 
                                           u_scatter_test,
                                           data_2=None,
                                          )

print(f"Final eval pde loss is {pde_loss/len(eval_data)}")


# Plot results
plt.figure()
plt.title('Test Loss')
plt.plot(test_loss_vec)
plt.show()

plt.figure()
plt.title('Magnitude')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=torch.abs(u_total))
plt.colorbar(sc)
plt.show()

plt.figure()
plt.title('Phase')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=torch.angle(u_total))
plt.colorbar(sc)
plt.show()

plt.figure()
plt.title('Magnitude Plane Wave')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=torch.abs(u_in))
plt.colorbar(sc)
plt.show()

plt.figure()
plt.title('Phase Plane Wave')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=torch.angle(u_in))
plt.colorbar(sc)
plt.show()
