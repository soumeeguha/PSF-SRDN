import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import torch

# Step 1: Define the 2D PSF (sinusoidal in x-direction, Gaussian in both x and y)
def sinusoidal_gaussian_psf_2d(size, k0, sigma_x, sigma_y):
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)

    # print(x, y)

    # h2: Sinusoidal + Gaussian decay in x
    psf_y = torch.sin(k0 * y) * torch.exp(-y**2 / (2 * sigma_y**2))
    
    
    # h1: Gaussian decay in y
    psf_x = torch.exp(-x**2 / (2 * sigma_x**2))


    # The final PSF is the product of h1(x) and h2(y)
    psf = torch.from_numpy(convolve(psf_x, psf_y))
    # print(np.shape(psf))
    # print((psf))
    psf /= torch.abs(psf).sum()  # Normalize the PSF
    return psf