import os
import argparse
import numpy as np
import pandas as pd
import torch

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.transforms import Compose

from psf_srdn.model import Unet
from psf_srdn.get_dataset import get_train_test_image_lists, get_Dataset, setup_dataloader
from psf_srdn.utils import setup_utils
from train import train_epochs
from evaluate import *


def main():
    args = parse_args()

    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    

    # PSF setup
    f0 = 10e6
    c = 1540
    k0 = torch.tensor(2 * np.pi * f0 / c)

    sigma_x = torch.linspace(args.sigma_x_start, args.sigma_x_end, args.sigma_steps)
    sigma_y = torch.linspace(args.sigma_y_start, args.sigma_y_end, args.sigma_steps)
    sigma_xy = torch.cat((sigma_x.unsqueeze(0), sigma_y.unsqueeze(0)), dim=0).to(device)

    # Dataset loading
    train_img, val_img = get_train_test_image_lists(args.root, folders=True)

    train_transform = Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    ds_train = get_Dataset(256, train_img, train_transform)
    ds_val = get_Dataset(256, val_img, test_transform)

    dataloader, _ = setup_dataloader(args.batch_size, num_worker_train=2, train_ds=ds_train, test_ds=ds_val)

    # Model

    timesteps = 200
    setup_utils(timesteps)

    model = Unet(
        dim=args.image_size,
        psf_dim=args.sigma_steps,
        timesteps=timesteps,
        channels=args.channels,
        dim_mults=(1, 2, 4)
    ).to(device)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    # Create output directories
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)

    # Train
    train_epochs(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_xy=sigma_xy,
        psf_size=args.psf_size,
        k0=k0,
        sigma_steps=args.sigma_steps,
        timesteps=timesteps,
        batch_size=args.batch_size,
        save_model_path=args.save_model_path
    )

    # Test 

    steps = list(range(0, timesteps))
    test_times = [step for step in steps if step % 4 == 3]
    os.makedirs(args.save_metrics_path, exist_ok=True)

    sigma_x_test = torch.tensor([2.0, 3.0, 4.0])
    sigma_y_test = torch.tensor([1.5, 2.5, 3.5])

    for i in range(len(sigma_y_test)):
        print(f'sigma_x: {sigma_x_test[i]}, sigma_y: {sigma_y_test[i]}')

        mse_score, psnr_score, ssim_score = evaluate_test_imgs(
            model=model,
            timesteps=timesteps,
            ds_val =ds_val,
            test_times=test_times,
            sigma_x_chosen=sigma_x_test[i],
            sigma_y_chosen=sigma_y_test[i],
            device=device,
            psf_size=args.psf_size,
            k0=k0,
            sigma_xy=sigma_xy,
            batch_size=8
        )

        eval_data = {
            'timestep': test_times,
            'MSE': mse_score,
            'PSNR': psnr_score,
            'SSIM': ssim_score,
            'sigma_x': sigma_x_test[i].item(),
            'sigma_y': sigma_y_test[i].item()
        }
        eval_df = pd.DataFrame(eval_data)
        eval_df.to_csv(f'{args.save_metrics_path}/sigmaX_{sigma_x_test[i]}_sigmaY_{sigma_y_test[i]}.csv', index=False)



def parse_args():
    parser = argparse.ArgumentParser(description="Train PSF-SRDN model with ultrasound data")

    # Dataset root
    parser.add_argument('--root', type=str, required=True, help='Path to image set')

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--image_size', type=int, default=64, help='Image size (cropped)')
    parser.add_argument('--channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--psf_size', type=int, default=3, help='PSF kernel size')
    parser.add_argument('--sigma_steps', type=int, default=50, help='Steps in sigma grid')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')

    # PSF sigma bounds
    parser.add_argument('--sigma_x_start', type=float, default=1.0)
    parser.add_argument('--sigma_x_end', type=float, default=4.0)
    parser.add_argument('--sigma_y_start', type=float, default=0.5)
    parser.add_argument('--sigma_y_end', type=float, default=3.5)

    # Device
    parser.add_argument('--cuda_id', type=int, default=0, help='GPU id to use')

    # Output paths
    parser.add_argument('--save_model_path', type=str, default='./saved_models/model.pth')
    parser.add_argument('--save_metrics_path', type=str, default='./metrics')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    main()
