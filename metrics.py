import torch
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure

def get_mse(input, output):
    squared_error = (input - output)**2
    return squared_error.mean()

def get_psnr(input, output):
    mse = get_mse(input, output)
    psnr = 10*(torch.log10((output.max()**2)/mse))
    return psnr

def get_ssim(input, output):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(torch.tensor(output), input)