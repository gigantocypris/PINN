import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
from collections import OrderedDict

"""
The sine layer is the basic building block of SIREN. This is a much more concise implementation than the one in the main code, as here, we aren't concerned with the baseline comparisons.
"""

class SineLayer(nn.Module):
    """
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the nonlinearity. Different signals may require different omega_0 in the first layer - this is a hyperparameter.

    If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, num_basis, two_d, num_hidden_layers=3, hidden_layer_width=64, outermost_linear=False,
                 first_omega_0=5.0, hidden_omega_0=5.0):
        super().__init__()

        in_features = 2 if two_d else 3
        out_features = num_basis*2
        self.num_basis = num_basis
        self.net = []
        self.net.append(SineLayer(in_features, hidden_layer_width,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(num_hidden_layers):
            self.net.append(SineLayer(hidden_layer_width, hidden_layer_width,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_layer_width, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_layer_width) / hidden_omega_0,
                                              np.sqrt(6 / hidden_layer_width) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_layer_width, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        output = torch.reshape(output, (-1,self.num_basis,2)) # last dimension is the real and imaginary parts
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output before final reshape, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations