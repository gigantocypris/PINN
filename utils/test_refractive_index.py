import numpy as np
import torch
import matplotlib.pyplot as plt
from physics import evaluate_refractive_index

sharpness_vec = np.arange(1,10,1)
n_inclusion=1.88
n_background=1.33
radius=3
sharpness=2
x = np.arange(-10,10,0.1)
y = np.zeros_like(x)
data = np.stack((x,y),axis=1)
data = torch.tensor(data).float()

plt.figure()

for sharpness in sharpness_vec:
    refractive_index = evaluate_refractive_index(data, n_background, n_inclusion, radius, sharpness)
    plt.plot(x, refractive_index, label="sharpness = " + str(sharpness))
    breakpoint()
plt.legend()
plt.savefig("refractive_index_line_plot.png")