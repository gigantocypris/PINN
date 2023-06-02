import torch
from torch.utils.data import DataLoader
from utils import create_data, NeuralNetwork, get_pde_loss, train, test, get_k0, create_plane_wave
import matplotlib.pyplot as plt
import numpy as np

# units are microns

# set parameters
batch_size = 100
num_basis = 200 # number of basis functions, N in pde-cl paper
use_pde_cl = True # use the partial differential equation constrained layer
wavelength = 2.1 # um
n_background = 1
use_cpu = False
epochs = 10

# set the training region
training_data_x_start = [-2,-2,-2]
training_data_x_end = [2,2,2]
training_data_x_step = [0.1,0.1,0.1]

# set the test region
offset = 0.05
test_data_x_start = [-2,-2,-2]
test_data_x_end = [2,2,2]
test_data_x_step = [0.5,0.5,0.5]

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
training_data = create_data(training_data_x_start, training_data_x_end, training_data_x_step, device)

# Training data to compute pde loss
# Training data is a list of coordinates
# This is only used if the linear system is underdetermined
if batch_size<num_basis:
    training_data_2 = create_data(np.array(training_data_x_start)+offset, 
                                np.array(training_data_x_end)+offset, training_data_x_step, device)

# Test data for validation of pde loss
# Test data is a list of coordinates
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

# PDE loss function
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

# Train the PINN
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
model = NeuralNetwork(num_basis).to(device)
model.load_state_dict(torch.load("model.pth"))


# Use loaded model to make predictions
model.eval()

# Visualize the PINN with list of coordinates
offset_eval = 0.03
eval_data = create_data(training_data_x_start+offset_eval, training_data_x_end+offset_eval, training_data_x_step, device)


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

eval_data = eval_data.cpu().numpy()
u_total = u_total.cpu().numpy()
u_in = u_in.cpu().numpy()

# Plot results
plt.figure()
plt.title('Test Loss')
plt.plot(test_loss_vec)
plt.show()

plt.figure()
plt.title('Magnitude')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=np.abs(u_total))
plt.colorbar(sc)
plt.show()

plt.figure()
plt.title('Phase')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=np.angle(u_total))
plt.colorbar(sc)
plt.show()

plt.figure()
plt.title('Magnitude Plane Wave')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=np.abs(u_in))
plt.colorbar(sc)
plt.show()

plt.figure()
plt.title('Phase Plane Wave')
sc = plt.scatter(x=eval_data[:,0],y=eval_data[:,1],c=np.angle(u_in))
plt.colorbar(sc)
plt.show()
