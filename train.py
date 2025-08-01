import torch
from psf_srdn.diffusion_equations import p_losses
from get_psf import sinusoidal_gaussian_psf_2d


def train_epochs(
    model, 
    dataloader, 
    optimizer, 
    scheduler, 
    device, 
    epochs,
    sigma_x, sigma_y, sigma_xy, psf_size, k0, sigma_steps,
    timesteps, batch_size, save_model_path
):
    time = torch.arange(0, timesteps, device=device)
    total_loss = []

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch.to(device)

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            sigma_x_idx = torch.randint(0, sigma_steps, (1,), device=device).long()
            sigma_y_idx = torch.randint(0, sigma_steps, (1,), device=device).long()

            psf = sinusoidal_gaussian_psf_2d(psf_size, k0, sigma_x[sigma_x_idx], sigma_y[sigma_y_idx])
            psf = torch.tensor(psf).to(device).unsqueeze(0).unsqueeze(0).float()

            loss = p_losses(model, batch, t, psf, sigma_xy, time, loss_type="l1")
            total_loss.append(loss.item())

            if step % 100 == 0:
                print(f"Epoch: {epoch}/{epochs} Step: {step} Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()

        scheduler.step()

    torch.save(model.state_dict(), save_model_path)
    print(f"Saved model to {save_model_path}")

