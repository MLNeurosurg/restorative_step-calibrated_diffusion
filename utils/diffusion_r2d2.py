import numpy as np
import torch
import copy
import random
import torch.nn.functional as F
from .beta_schedulers import linear_beta_schedule, cosine_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
from tqdm.auto import tqdm
from scipy.linalg import dft
from data.transforms import reverse_transforms
from numpy.linalg import inv
"""
IDEA: Randomly change upscale or downscale the beta schedule
Ways to make the denoising less deterministic

One problem that we will need to address is how to handle variable amounts of:
1) Noise in the image, so how far back to go in the diffusion
2) How to vary the amount of the noise for each example at each timestep
3) Can we introduce other data corruptions in the image

The denoising process must also learn to restore the images

Combine generative and the image restoration >> diffusion process

"""


class Diffusion(object):
    def __init__(self,
                 restore_timesteps=100,
                 timesteps=200,
                 beta_schedule='linear',
                 image_size=300):

        self.timesteps = timesteps
        self.restore_timesteps = restore_timesteps
        assert self.restore_timesteps <= self.timesteps
        self.image_size = image_size

        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif beta_schedule == 'quadratic':
            self.betas = quadratic_beta_schedule(timesteps=self.timesteps)
        elif beta_schedule == 'sigmoid':
            self.betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        else:
            raise ValueError(
                'Beta schedule not supported. Use linear, cosine, quadratic, or sigmoid.'
            )

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0),
                                         value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. -
                                                        self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (
            1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        #### Get Images ####
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size,
                           *((1, ) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property)
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t,
                                             x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_noisy_image(self, x_start, t):
        # add noise
        x_noisy = self.q_sample(x_start, t=t)

        # turn back into PIL image
        noisy_image = reverse_transforms(x_noisy.squeeze())

        return noisy_image

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index,origfeat):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t,
                                                x.shape)
            noise = torch.randn_like(x)

            return (model_mean + torch.sqrt(posterior_variance_t) * noise)*0.95+origfeat*0.05

    @torch.no_grad()
    # def p_sample_loop(self, model, shape):
    def p_sample_loop(self, model, x_input, shape, begin, train=False,step_limit=None,origfeat=None):

        device = next(model.parameters()).device

        if train:
            img = self.q_sample(x_input,
                                t=torch.full((1, ),
                                             self.restore_timesteps,
                                             device=device))
        else:
            img = x_input
        # img = torch.randn(shape, device=device)

        imgs = [img.detach().cpu().numpy()]
        for i in tqdm(reversed(range(0, begin)),
                      desc='sampling loop time step',
                      total=begin):
            img = self.p_sample(
                model, img,
                torch.full((1, ), i, device=device, dtype=torch.long), i,origfeat=origfeat)
            imgs.append(img.detach().cpu().numpy())
            if step_limit is not None and i == begin - step_limit:
                return imgs
        return imgs
    @torch.no_grad()
    def getfeat(self,orig):
        f = dft(256)
        p = np.zeros((256,256))
        for i in range(128):
            p[i][i] = 1
        mult = np.matmul(p,f)
        mult = np.matmul(inv(f),mult)
        res = np.matmul(mult,orig.cpu().numpy())
        return torch.FloatTensor(res).to(orig.device)


    @torch.no_grad()
    # def sample(self, model, batch_size=16, channels=3):
    def sample(self, model, x_input, batch_size=16, channels=3, train=False,begin=None, step_limit=None, orig=None):
        if begin is None:
            begin = self.restore_timesteps
        origfeat = self.getfeat(orig)
        return self.p_sample_loop(model,
                                  x_input,
                                  shape=(batch_size, channels, self.image_size,
                                         self.image_size),begin=begin,step_limit=step_limit,
                                  train=train,origfeat=origfeat)
