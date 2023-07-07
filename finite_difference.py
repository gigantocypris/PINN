"""Finite Difference Solution for Mie scattering problem in 2D"""

import numpy as np
import matplotlib.pyplot as plt

# Inputs
x_start = [-5.0,-5.0]
x_end = [5.0,5.0]
x_step = [0.05,0.05]
wavelength = 1.0 # wavelength in free space
pml_grid_points = 5
n_background = 1.33
radius = 3.0

# Calculated inputs
k_0 = 2*np.pi/wavelength # wavenumber in free space
x = np.arange(x_start[0],x_end[0],x_step[0])
y = np.arange(x_start[1],x_end[1],x_step[1])
domain_size_x = len(x)
domain_size_y = len(y)
dx = x_step[0]
dy = x_step[1]

# Construct the grid

xm, ym = np.meshgrid(x, y, indexing='ij')
data = np.stack([xm,ym],axis=-1) # x_size x y_size x 2

# Construct the refractive index
refractive_index = np.where(np.sum(data**2,axis=-1)<radius**2, 1.88, n_background)

# Construct the plane wave
amplitude=1
tilt_theta=np.pi/2
k = n_background*k_0
kx = k*np.cos(tilt_theta)
kz = np.sqrt(k**2-kx**2)
k = np.array([[[kx, kz]]])
plane_wave = amplitude*np.exp(-1j*(np.sum(k*data, axis=-1)))

# Construct the source
E = -(refractive_index**2-n_background**2)*k_0**2*plane_wave

# Helper function for PML

def e_i(i, domain_size_i, pml_grid_points, a_0=5):
    if i < pml_grid_points or i > domain_size_i-pml_grid_points-1:
        dist_to_edge = min(i,domain_size_i-i)
    else:
        dist_to_edge = 0
    sigma = a_0*(dist_to_edge/pml_grid_points)**2
    
    e = 1-1j*sigma

    d_dist_d_x = 0
    if i < pml_grid_points:
        d_dist_d_x = 2*i
    elif i > domain_size_i-pml_grid_points-1:
        d_dist_d_x = -2*(domain_size_i-i-1)

    coeff = -1j*a_0/pml_grid_points**2
    
    return e, coeff, d_dist_d_x

# Construct finite difference matrix A
A = np.zeros([domain_size_x * domain_size_y, domain_size_x * domain_size_y], dtype=np.complex128)
for i in range(0, domain_size_x):
    for j in range(0, domain_size_y):
        n = i * domain_size_y + j  # Current grid point index
        refractive_index_n = refractive_index[i, j]  # Refractive index at the current grid point

        if i < pml_grid_points or i > domain_size_x - pml_grid_points - 1 or j < pml_grid_points or j > domain_size_y - pml_grid_points - 1:
            # PML region
            e_x, coeff_x, d_dist_d_x  = e_i(i, domain_size_x, pml_grid_points)
            e_y, coeff_y, d_dist_d_y = e_i(j, domain_size_y, pml_grid_points)
            A[n, n] = (e_y/e_x)*(-2 / dx**2) - (e_x/e_y) * (2 / dy**2) + e_x*e_y*k_0**2 * refractive_index_n**2
            
            try:
                A[n, (i - 1) * domain_size_y + j] = (e_y/e_x) * (1 / dx**2) - e_y*coeff_x*d_dist_d_x/(-e_x**2) * (1 / 2*dx)
            except IndexError:
                pass

            try:
                A[n, (i + 1) * domain_size_y + j] = (e_y/e_x) * (1 / dx**2) + e_y*coeff_x*d_dist_d_x/(-e_x**2) * (1 / 2*dx)
            except IndexError:
                pass

            try:
                A[n, i * domain_size_y + (j - 1)] = (e_x/e_y) * (1 / dy**2) - e_x*coeff_y*d_dist_d_y/(-e_y**2) * (1 / 2*dy)
            except IndexError:
                pass

            try:
                A[n, i * domain_size_y + (j + 1)] = (e_x/e_y) * (1 / dy**2) + e_x*coeff_y*d_dist_d_y/(-e_y**2) * (1 / 2*dy)
            except IndexError:
                pass

        else:
            # Non-PML region
            A[n, n] = -2 / dx**2 - 2 / dy**2 + k_0**2 * refractive_index_n**2
            try:
                A[n, (i - 1) * domain_size_y + j] = 1 / dx**2
            except IndexError:
                pass
            try:
                A[n, (i + 1) * domain_size_y + j] = 1 / dx**2
            except IndexError:
                pass
            try:
                A[n, i * domain_size_y + (j - 1)] = 1 / dy**2
            except IndexError:
                pass
            try:
                A[n, i * domain_size_y + (j + 1)] = 1 / dy**2
            except IndexError:
                pass



# Solve the system of equations
U_s = np.linalg.solve(A, E.flatten())
U_s = U_s.reshape(domain_size_x, domain_size_y)


# Plotting the results
plt.figure()
plt.imshow(np.abs(U_s), cmap='hot', origin='lower', extent=[0, dx * (domain_size_x - 1), 0, dy * (domain_size_y - 1)])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scattered Field Distribution')
plt.colorbar()
plt.savefig('scattered_field.png')
plt.show()
