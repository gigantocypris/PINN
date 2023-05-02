import numpy as np
import torch
from torch import nn

def create_data(min_vals, max_vals, spacings, device):
    data_x = torch.arange(min_vals[0],max_vals[0],spacings[0],device=device)
    data_y = torch.arange(min_vals[1],max_vals[1],spacings[1],device=device)
    data_z = torch.arange(min_vals[2],max_vals[2],spacings[2],device=device)
    data_xm, data_ym, data_zm = torch.meshgrid(data_x, data_y, data_z, indexing='ij')
    data = torch.stack((data_xm, data_ym, data_zm), dim=3)
    data = torch.reshape(data, (-1,3))
    return data

class NeuralNetwork(nn.Module):
    def __init__(self, num_basis):
        super().__init__()
        self.num_basis = num_basis
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_basis*2),
            nn.Tanh(),
        )

    def forward(self, x):
        u_scatter = self.linear_relu_stack(x)
        u_scatter = torch.reshape(u_scatter, (-1,self.num_basis,2))
        return u_scatter

def evalulate_refractive_index(data,
                               n_background, 
                               radius=0.5, # um
                              ):
    return torch.where(torch.sum(data**2,dim=1)<radius**2, 1.3, n_background)

def get_k0(wavelength):
    return 2*np.pi/wavelength

def create_plane_wave(data, 
                      wavelength,
                      n_background,
                      device,
                      amplitude=1,
                      tilt_theta=np.pi/2, # radians, angle from the xy plane, np.pi/2 is in the z-direction
                      tilt_phi=0, # radians, angle in the xy plane
                     ):
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

def transform_linear_pde(data, 
                         k0,
                         n_background,
                         u_scatter,
                         model,
                         device,
                        ):

    hess = torch.vmap(torch.func.hessian(model, argnums=0),in_dims=(0))(data)
    refractive_index = evalulate_refractive_index(data, n_background) 
    
    du_scatter_xx = torch.squeeze(hess[:,:,:,:,0,0], dim=1)
    du_scatter_yy = torch.squeeze(hess[:,:,:,:,1,1], dim=1)
    du_scatter_zz = torch.squeeze(hess[:,:,:,:,2,2], dim=1)
    
    du_scatter_xx_complex = du_scatter_xx[:,:,0]+1j*du_scatter_xx[:,:,1]
    du_scatter_yy_complex = du_scatter_yy[:,:,0]+1j*du_scatter_yy[:,:,1]
    du_scatter_zz_complex = du_scatter_zz[:,:,0]+1j*du_scatter_zz[:,:,1]
    u_scatter_complex = u_scatter[:,:,0]+1j*u_scatter[:,:,1]
    
    linear_pde = du_scatter_xx_complex+du_scatter_yy_complex+du_scatter_zz_complex+k0**2*torch.unsqueeze(refractive_index,dim=1)**2*u_scatter_complex
    return linear_pde, refractive_index, u_scatter_complex

def transform_affine_pde(wavelength,
                         data,
                         k0,
                         n_background,
                         u_scatter,
                         model,
                         device,
                         ):
    
    linear_pde, refractive_index, u_scatter_complex = transform_linear_pde(data,
                                                                           k0,
                                                                           n_background,
                                                                           u_scatter,
                                                                           model,
                                                                           device,
                                                                          )
    u_in = create_plane_wave(data, 
                             wavelength,
                             n_background,
                             device)
    f = k0**2*(refractive_index**2 - n_background**2)*u_in
    f=torch.unsqueeze(f,dim=1)
    return linear_pde, f, u_scatter_complex, u_in

def get_pde_loss(data, 
                 wavelength,
                 n_background,
                 u_scatter,
                 model,
                 device,
                 use_pde_cl,
                 data_2=None,
                 ):

    k0 = get_k0(wavelength)

    linear_pde, f, u_scatter_complex, u_in = \
    transform_affine_pde(wavelength,
                        data,
                        k0,
                        n_background,
                        u_scatter,
                        model,
                        device,
                        )


    if use_pde_cl:
        w = torch.linalg.lstsq(linear_pde, f, driver='gelsd').solution
        linear_pde_combine = torch.matmul(linear_pde,w)
        u_scatter_complex_combine = torch.matmul(u_scatter_complex,w)
        u_scatter_complex_combine = torch.squeeze(u_scatter_complex_combine, dim=1)
    else:
        linear_pde_combine = linear_pde[:,0]
        linear_pde_combine = torch.unsqueeze(linear_pde_combine,dim=1)
        u_scatter_complex_combine = u_scatter_complex[:,0]
        
    # combine the scattered field with the incident field
    u_total = u_scatter_complex_combine+u_in

    if data_2 is not None:
        linear_pde, f, _, _ = \
        transform_affine_pde(wavelength,
                            data_2,
                            k0,
                            n_background,
                            u_scatter,
                            model,
                            device,
                            )
        linear_pde_combine = torch.matmul(linear_pde,w)

    pde = linear_pde_combine-f
    pde = torch.squeeze(pde, dim=1)
    pde_loss = torch.sum(torch.abs(pde))
    return pde_loss, u_total, u_scatter_complex_combine

def train(dataloader, 
          dataloader_2,
          model, 
          loss_fn, 
          optimizer,
          device,
          ):
    
    if dataloader_2 is not None:
        dataloader_2_iter = iter(dataloader_2)
    size = len(dataloader.dataset)
    model.train()
    total_examples_finished = 0
    for data in dataloader:
        if dataloader_2 is not None:
            data_2 = next(dataloader_2_iter)
            data_2 = data_2.to(device)
        else:
            data_2 = None
        data = data.to(device)
        # Compute prediction error
        u_scatter = model(data)
        pde_loss, _, _ = loss_fn(data, 
                                 u_scatter,
                                 data_2,
                                )
        pde_loss = pde_loss/len(data)
        # Backpropagation
        optimizer.zero_grad()
        pde_loss.backward()
        optimizer.step()
        total_examples_finished += len(data)
        pde_loss = pde_loss.item()
        print(f"loss: {pde_loss:>7f}  [{total_examples_finished:>5d}/{size:>5d}]")

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
            pde_loss, _, _ = loss_fn(data, 
                                     u_scatter,
                                     data_2=None,
                                    )
            test_loss += pde_loss.item()
            
    test_loss /= size
    print(f"Avg test loss: {test_loss:>8f}")
    return test_loss