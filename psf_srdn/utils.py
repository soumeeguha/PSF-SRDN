import torch
from torch.nn.modules import linear
import torch.nn.functional as F
from psf_srdn.beta_schedules import *

alphas = None 

def setup_utils(timesteps):
    global alphas
    alphas = linear_beta_schedule(timesteps)
    if not isinstance(alphas, torch.Tensor):
        alphas = torch.tensor(alphas, dtype=torch.float32)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)