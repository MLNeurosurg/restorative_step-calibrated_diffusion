import torch
import torch.nn.functional as F


def p_losses(denoise_model, diffusion, x_start, t, noise=None, loss_type="l1"):

    if noise is None:
        # returns tensor of same size as x_start with mean 0 and std 1
        noise = torch.randn_like(x_start)

    # x_start are the images
    # forward diffusion image
    x_noisy = diffusion.q_sample(x_start=x_start, t=t, noise=noise)
    # the predicted noise within the x_noisy at time t
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
