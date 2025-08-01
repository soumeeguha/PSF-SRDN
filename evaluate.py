import torch 
from torch.utils.data import DataLoader
from psf_srdn.diffusion_equations import *
from get_psf import sinusoidal_gaussian_psf_2d
from metrics import *


def evaluate_test_imgs(
        model, timesteps, ds_val, test_times, sigma_x_chosen, sigma_y_chosen, device, 
        psf_size, k0, sigma_xy, batch_size=1
):
    test_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last = True)
    time = torch.arange(0, timesteps, device=device)

    mse_score, psnr_score, ssim_score = [], [], []

    for test_time in test_times:
        mse, psnr, ssim = 0, 0, 0
        num_batch = 0
        for batch in test_dataloader:
            t = torch.tensor([test_time], device=device).long()
            psf = sinusoidal_gaussian_psf_2d(psf_size, k0, sigma_x_chosen, sigma_y_chosen)
            psf = psf.to(device).unsqueeze(0).unsqueeze(0).float()

            x_noisy = q_sample_multiplicative(x_start=batch.to(device), t=t, psf=psf)

            samples = sample(model, time, sigma_xy, noisy = x_noisy, device = device)

            mse += get_mse(batch, samples[0])
            psnr += get_psnr(batch, samples[0])
            ssim += get_ssim(batch, samples[0])

            num_batch += 1

        mse_score.append(mse.item() / num_batch)
        psnr_score.append(psnr.item() / num_batch)
        ssim_score.append(ssim.item() / num_batch)

    return mse_score, psnr_score, ssim_score