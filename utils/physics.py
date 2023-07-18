import numpy as np
import torch

def evaluate_refractive_index(data,
                               n_background, 
                               n_inclusion=1.88, # refractive index of the inclusion
                               radius=3, # um
                               sharpness=9, # higher value means sharper boundary
                              ):
    """evalulate the refractive index at the data points for spherical/cylinderical dielectric"""
    # refractive_index = torch.where(torch.sum(data**2,dim=1)<radius**2, 1.88, n_background)

    dist = torch.sqrt(torch.sum(data**2,dim=1))
    refractive_index = (n_inclusion-n_background)*torch.sigmoid(sharpness*(-dist+radius))+n_background
    return refractive_index


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

# Helper function for PML

def get_e_i(i, domain_size_i, L_pml, a_0=0.25):
    """
    sigma in this code = sigma_siren_paper/omega
    domain_size_i and L_pml are in the same units
    """
    if i < L_pml or i > domain_size_i-L_pml:
        dist_to_edge = min(i,domain_size_i-i)
    else:
        dist_to_edge = 0
    sigma = a_0*(dist_to_edge/L_pml)**2
    
    e = 1-1j*sigma

    d_dist_squared_d_x = 0
    if i < L_pml:
        d_dist_squared_d_x = 2*i
    elif i > domain_size_i-L_pml:
        d_dist_squared_d_x = -2*(domain_size_i-i)

    coeff = -1j*a_0/L_pml**2
    
    return e, coeff, d_dist_squared_d_x

def get_e_i_batched(i_vec, domain_size_i, L_pml, a_0=0.25):
    """
    sigma in this code = sigma_siren_paper/omega
    domain_size_i and L_pml are in the same units
    """

    dist_to_edge = torch.zeros_like(i_vec)
    dist_to_edge[torch.where(i_vec < L_pml)]=i_vec[torch.where(i_vec < L_pml)]
    dist_to_edge[torch.where(i_vec > domain_size_i-L_pml)]=domain_size_i-i_vec[torch.where(i_vec > domain_size_i-L_pml)]

    sigma = a_0*(dist_to_edge/L_pml)**2
    
    e = 1-1j*sigma

    d_dist_squared_d_x = torch.zeros_like(i_vec)
    d_dist_squared_d_x[torch.where(i_vec < L_pml)]=2*i_vec[torch.where(i_vec < L_pml)]
    d_dist_squared_d_x[torch.where(i_vec > domain_size_i-L_pml)]=-2*domain_size_i+2*i_vec[torch.where(i_vec > domain_size_i-L_pml)]   

    coeff = -1j*a_0/L_pml**2

    return e, coeff, d_dist_squared_d_x

def transform_linear_pde(data, 
                         k0,
                         n_background,
                         u_scatter,
                         model,
                         two_d,
                         device,
                         domain_size_x,
                         domain_size_z,
                         L_pml_x,
                         L_pml_z,
                         a_0=0.25,
                         use_vmap=True,
                        ):
    '''Get the right hand side of the PDE (del**2 + n**2*k0**2)*u_scatter = -(n**2-n_background**2)*k0**2*u_in))'''
    hess_fn = torch.func.hessian(model, argnums=0)
    if use_vmap:
        hess = torch.vmap(hess_fn,in_dims=(0))(data) # hessian
    else:
        hess = []
        for i in range(data.size(0)):
            hess_i = hess_fn(data[i])
            hess.append(hess_i)
        # Concatenate the outputs to form a single tensor
        hess = torch.stack(hess, dim=0) 

    # get the Jacobian for computing the right hand side of the PDE with PML boundary conditions
    # Right hand side of PDE with the PML boundary is (left hand side is unchanged): 
    # (d/dx eps_y/eps_x d/dx)(u_scatter) + (d/dy eps_x/eps_y d/dy)(u_scatter) + eps_x*eps_y*n**2*k0**2*u_scatter
    jacobian_fn = torch.func.jacfwd(model, argnums=0)
    if use_vmap:
        jacobian = torch.vmap(jacobian_fn,in_dims=(0))(data) # jacobian
    else:
        jacobian = []
        for i in range(data.size(0)):
            jacobian_i = jacobian_fn(data[i])
            jacobian.append(jacobian_i)
        # Concatenate the outputs to form a single tensor
        jacobian = torch.stack(jacobian, dim=0) 
    
    du_scatter_x = torch.squeeze(jacobian[:,:,:,:,0], dim=1)
    if two_d:
        du_scatter_z = torch.squeeze(jacobian[:,:,:,:,1], dim=1)
    else:
        du_scatter_y = torch.squeeze(jacobian[:,:,:,:,1], dim=1)
        du_scatter_z = torch.squeeze(jacobian[:,:,:,:,2], dim=1)

    du_scatter_x_complex = du_scatter_x[:,:,0]+1j*du_scatter_x[:,:,1]
    if not two_d:
        du_scatter_y_complex = du_scatter_y[:,:,0]+1j*du_scatter_y[:,:,1]
    du_scatter_z_complex = du_scatter_z[:,:,0]+1j*du_scatter_z[:,:,1]

    e_x, coeff_x, d_dist_squared_d_x = get_e_i_batched(data[:,0], domain_size_x, L_pml_x, a_0=a_0)
    e_z, coeff_z, d_dist_squared_d_z = get_e_i_batched(data[:,1], domain_size_z, L_pml_z, a_0=a_0)
    e_x = torch.unsqueeze(e_x,dim=1)
    e_z = torch.unsqueeze(e_z,dim=1)    
    d_dist_squared_d_x = torch.unsqueeze(d_dist_squared_d_x,dim=1)
    d_dist_squared_d_z = torch.unsqueeze(d_dist_squared_d_z,dim=1)

    refractive_index = evaluate_refractive_index(data, n_background) 
    
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
        # linear_pde = du_scatter_xx_complex+du_scatter_zz_complex+k0**2*torch.unsqueeze(refractive_index,dim=1)**2*u_scatter_complex
        linear_pde = (-1)*e_z*du_scatter_x_complex*(e_x)**(-2)*coeff_x*d_dist_squared_d_x + \
                     (e_z/e_x) * (du_scatter_xx_complex) + \
                     (-1)*e_x*du_scatter_z_complex*(e_z)**(-2)*coeff_z*d_dist_squared_d_z + \
                     (e_x/e_z) * (du_scatter_zz_complex) + \
                     e_x*e_z*k0**2*torch.unsqueeze(refractive_index,dim=1)**2*u_scatter_complex
    else:
        linear_pde = du_scatter_xx_complex+du_scatter_yy_complex+du_scatter_zz_complex+k0**2*torch.unsqueeze(refractive_index,dim=1)**2*u_scatter_complex
        raise NotImplementedError('Boundary condition for 3 dimensions not implemented')
    return linear_pde, refractive_index, u_scatter_complex

def transform_affine_pde(args,
                         data,
                         k0,
                         u_scatter,
                         model,
                         device,
                         ):
    '''Get the right and left hand side of the PDE (del**2 + n**2*k0**2)*u_scatter = -(n**2-n_background**2)*k0**2*u_in))'''

    domain_size_x = args.data_x_end[0]-args.data_x_start[0]
    domain_size_z = args.data_x_end[1]-args.data_x_start[1]
    L_pml_x = args.pml_thickness[0]
    L_pml_z = args.pml_thickness[1]
    # get the right hand side of the PDE
    linear_pde, refractive_index, u_scatter_complex = transform_linear_pde(data,
                                                                           k0,
                                                                           args.n_background,
                                                                           u_scatter,
                                                                           model,
                                                                           args.two_d,
                                                                           device,
                                                                           domain_size_x,
                                                                           domain_size_z,
                                                                           L_pml_x,
                                                                           L_pml_z,
                                                                          )
    if args.two_d:
        u_in = create_plane_wave_2d(data, 
                                    args.wavelength,
                                    args.n_background,
                                    device)
    else:
        u_in = create_plane_wave_3d(data, 
                                    args.wavelength,
                                    args.n_background,
                                    device)
    
    # get the left hand side of the PDE
    f = -k0**2*(refractive_index**2 - args.n_background**2)*u_in
    f = torch.unsqueeze(f,dim=1)
    return linear_pde, f, u_scatter_complex, u_in, refractive_index

def get_pde_loss(args,
                 data, 
                 u_scatter,
                 model,
                 device,
                 data_2=None,
                 w=None,
                 ):

    k0 = get_k0(args.wavelength)

    linear_pde, f, u_scatter_complex, u_in, refractive_index = \
    transform_affine_pde(args,
                        data,
                        k0,
                        u_scatter,
                        model,
                        device,
                        )


    if args.use_pde_cl:
        if w is None:
            w = torch.linalg.lstsq(linear_pde, f, driver='gels').solution
        linear_pde_combine = torch.matmul(linear_pde,w)
        u_scatter_complex_combine = torch.matmul(u_scatter_complex,w)
        u_scatter_complex_combine = torch.squeeze(u_scatter_complex_combine, dim=1)
        # breakpoint()
    else:
        linear_pde_combine = linear_pde[:,0]
        linear_pde_combine = torch.unsqueeze(linear_pde_combine,dim=1)
        u_scatter_complex_combine = u_scatter_complex[:,0]
        
    # combine the scattered field with the incident field
    u_total = u_scatter_complex_combine+u_in

    # if underdetermined, use the second dataset to get the loss
    if data_2 is not None:
        linear_pde, f, _, _, _ = \
        transform_affine_pde(data_2,
                            k0,
                            u_scatter,
                            model,
                            device,
                            )
        linear_pde_combine = torch.matmul(linear_pde,w)

    pde = linear_pde_combine-f
    pde = torch.squeeze(pde, dim=1)
    pde_loss = torch.sum(torch.abs(pde)**2)
    return pde_loss, u_total, u_scatter_complex_combine, refractive_index, w
