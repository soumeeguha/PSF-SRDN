import torch
from psf_srdn.utils import *
import psf_srdn.utils as utils
import torch.nn.functional as F

def q_sample_multiplicative(x_start, t, psf, noise = None):

    if noise is None:
        noise = torch.randn_like(x_start)

    alphas = utils.alphas
    alpha_t = extract(alphas, t, x_start.shape)
    noisy = x_start*(1 + alpha_t * noise)
    
    return F.conv2d(noisy, psf, padding = 1)


def p_losses(denoise_model, x_start, t, psf, sigma_xy, time, noise=None, loss_type="l1"):
    
    
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample_multiplicative(x_start, t, psf, noise = noise)
    
    predicted_x_start = denoise_model(x_noisy, time, sigma_xy)

    
    if loss_type == 'l1':
        loss = F.l1_loss(x_start, predicted_x_start)
    elif loss_type == 'l2':
        loss = F.mse_loss(x_start, predicted_x_start)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(x_start, predicted_x_start)
    else:
        raise NotImplementedError()
    
    return loss

@torch.no_grad()
def p_sample(model, time, sigma_xy, x):

    predicted_x_start = model(x, time, sigma_xy)
    
    return predicted_x_start
    


# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, time, sigma_xy, image, device):
    device = next(model.parameters()).device

    img = image.to(device)
    imgs = []
    
    img = p_sample(model, time, sigma_xy, img)
    imgs.append(img.cpu().numpy())
    return imgs

def sample(model, time, sigma_xy, noisy, device):
    return p_sample_loop(model, time, sigma_xy, image = noisy, device=device)
