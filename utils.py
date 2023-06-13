"""Tests for AoC 1, 2022: Calorie Counting."""

import numpy as np
import torch
from torch import nn
from random import Random

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

        for ind,frac in enumerate(sizes):
            part_len = int(frac*data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class NeuralNetwork(nn.Module):
    def __init__(self, num_basis, two_d):
        super().__init__()
        input_dim = 2 if two_d else 3
        self.num_basis = num_basis
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_basis*2),
            nn.Tanh(),
        )

    def forward(self, x):
        u_scatter = self.linear_relu_stack(x)
        u_scatter = torch.reshape(u_scatter, (-1,self.num_basis,2)) # last dimension is the real and imaginary parts
        return u_scatter

def evalulate_refractive_index(data,
                               n_background, 
                               radius=3, # um
                              ):
    """evalulate the refractive index of at the data points for spherical dielectric"""
    return torch.where(torch.sum(data**2,dim=1)<radius**2, 1.88, n_background)


def get_k0(wavelength):
    """get free space wavenumber"""
    return 2*np.pi/wavelength


def create_plane_wave_3d(data, 
                         wavelength,
                         n_background,
                         device,
                         amplitude=1,
                         tilt_theta=np.pi/2, # radians, angle from the xy plane, np.pi/2 is in the z-direction
                         tilt_phi=0, # radians, angle in the xy plane
                        ):
    """create free space plane wave u_in"""
    """z is the direction of propagation"""
    """x into the page, y up, z to the right"""
    k0 = get_k0(wavelength)
    k = n_background*k0
    k_xy = k*np.cos(tilt_theta)
    kx = k_xy*np.cos(tilt_phi)
    ky = k_xy*np.sin(tilt_phi)
    kz = np.sqrt(k**2-k_xy**2)
    k = torch.tensor([[kx, ky, kz]], dtype=torch.cfloat, device=device)
    return(amplitude*torch.exp(-1j*(torch.sum(k*data, dim=1))))

def create_plane_wave_2d(data, 
                         wavelength,
                         n_background,
                         device,
                         amplitude=1,
                         tilt_theta=np.pi/2, # radians, angle from the xy plane, np.pi/2 is in the z-direction
                        ):
    """create free space plane wave u_in"""
    """z is the direction of propagation"""
    """x is up, z to the right"""
    k0 = get_k0(wavelength)
    k = n_background*k0
    kx = k*np.cos(tilt_theta)
    kz = np.sqrt(k**2-kx**2)
    k = torch.tensor([[kx, kz]], dtype=torch.cfloat, device=device)
    return(amplitude*torch.exp(-1j*(torch.sum(k*data, dim=1))))

def transform_linear_pde(data, 
                         k0,
                         n_background,
                         u_scatter,
                         model,
                         two_d,
                        ):
    '''Get the right hand side of the PDE (del**2 + n**2*k0**2)*u_scatter = -(n**2-n_background**2)*k0**2*u_in))'''
    hess_fn = torch.func.hessian(model, argnums=0)
    hess = torch.vmap(hess_fn,in_dims=(0))(data) # hessian
    refractive_index = evalulate_refractive_index(data, n_background) 
    
    du_scatter_xx = torch.squeeze(hess[:,:,:,:,0,0], dim=1)

    if two_d:
        du_scatter_zz = torch.squeeze(hess[:,:,:,:,1,1], dim=1)
    else:
        du_scatter_yy = torch.squeeze(hess[:,:,:,:,1,1], dim=1)
        du_scatter_zz = torch.squeeze(hess[:,:,:,:,2,2], dim=1)
    
    du_scatter_xx_complex = du_scatter_xx[:,:,0]+1j*du_scatter_xx[:,:,1]
    if not two_d:
        du_scatter_yy_complex = du_scatter_yy[:,:,0]+1j*du_scatter_yy[:,:,1]
    du_scatter_zz_complex = du_scatter_zz[:,:,0]+1j*du_scatter_zz[:,:,1]
    u_scatter_complex = u_scatter[:,:,0]+1j*u_scatter[:,:,1]
    
    if two_d:
        linear_pde = du_scatter_xx_complex+du_scatter_zz_complex+k0**2*torch.unsqueeze(refractive_index,dim=1)**2*u_scatter_complex
    else:
        linear_pde = du_scatter_xx_complex+du_scatter_yy_complex+du_scatter_zz_complex+k0**2*torch.unsqueeze(refractive_index,dim=1)**2*u_scatter_complex
    return linear_pde, refractive_index, u_scatter_complex

def transform_affine_pde(wavelength,
                         data,
                         k0,
                         n_background,
                         u_scatter,
                         model,
                         device,
                         two_d,
                         ):
    '''Get the right and left hand side of the PDE (del**2 + n**2*k0**2)*u_scatter = -(n**2-n_background**2)*k0**2*u_in))'''

    # get the right hand side of the PDE
    linear_pde, refractive_index, u_scatter_complex = transform_linear_pde(data,
                                                                           k0,
                                                                           n_background,
                                                                           u_scatter,
                                                                           model,
                                                                           two_d,
                                                                          )
    if two_d:
        u_in = create_plane_wave_2d(data, 
                                    wavelength,
                                    n_background,
                                    device)
    else:
        u_in = create_plane_wave_3d(data, 
                                    wavelength,
                                    n_background,
                                    device)
    
    # get the left hand side of the PDE
    f = -k0**2*(refractive_index**2 - n_background**2)*u_in
    f = torch.unsqueeze(f,dim=1)
    return linear_pde, f, u_scatter_complex, u_in, refractive_index

def get_pde_loss(data, 
                 wavelength,
                 n_background,
                 u_scatter,
                 model,
                 device,
                 use_pde_cl,
                 two_d,
                 data_2=None,
                 ):

    k0 = get_k0(wavelength)

    linear_pde, f, u_scatter_complex, u_in, refractive_index = \
    transform_affine_pde(wavelength,
                        data,
                        k0,
                        n_background,
                        u_scatter,
                        model,
                        device,
                        two_d,
                        )


    if use_pde_cl:
        w = torch.linalg.lstsq(linear_pde, f, driver='gels').solution
        linear_pde_combine = torch.matmul(linear_pde,w)
        u_scatter_complex_combine = torch.matmul(u_scatter_complex,w)
        u_scatter_complex_combine = torch.squeeze(u_scatter_complex_combine, dim=1)
    else:
        linear_pde_combine = linear_pde[:,0]
        linear_pde_combine = torch.unsqueeze(linear_pde_combine,dim=1)
        u_scatter_complex_combine = u_scatter_complex[:,0]
        
    # combine the scattered field with the incident field
    u_total = u_scatter_complex_combine+u_in

    # if underdetermined, use the second dataset to get the loss
    if data_2 is not None:
        linear_pde, f, _, _, _ = \
        transform_affine_pde(wavelength,
                            data_2,
                            k0,
                            n_background,
                            u_scatter,
                            model,
                            device,
                            two_d,
                            )
        linear_pde_combine = torch.matmul(linear_pde,w)

    pde = linear_pde_combine-f
    pde = torch.squeeze(pde, dim=1)
    pde_loss = torch.sum(torch.abs(pde))
    return pde_loss, u_total, u_scatter_complex_combine, refractive_index

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
        data = data.to(device)
        rand_1 = jitter*(torch.rand(data.shape, dtype=dtype, device=device) - 0.5)
        if dataloader_2 is not None:
            data_2 = next(dataloader_2_iter)
            rand_2 = jitter*(torch.rand(data_2.shape, dtype=dtype, device=device) - 0.5)
            data_2 = data_2.to(device)
            data_2 += rand_2
        else:
            data_2 = None
        
        
        data += rand_1
        # Compute prediction error
        u_scatter = model(data)
        pde_loss, _, _, _ = loss_fn(data, 
                                    u_scatter,
                                    data_2.to(device) if data_2 is not None else None,
                                   )
        pde_loss = pde_loss/len(data)
        # Backpropagation
        optimizer.zero_grad()
        pde_loss.backward()
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
            pde_loss, _, _, _ = loss_fn(data, 
                                     u_scatter,
                                     data_2=None,
                                    )
            test_loss += pde_loss.item()
            
    test_loss /= size
    print(f"Avg test loss: {test_loss:>8f}")
    return test_loss

