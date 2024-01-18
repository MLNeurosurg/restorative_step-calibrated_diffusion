"""
https://github.com/lucidrains/imagen-pytorch
"""

import torch
from imagen_pytorch import Unet, Imagen, SRUnet256, SRUnet1024, ImagenTrainer
from torchsummary import summary
import sys


# dim_mults = (1, 2, 4, 8)
# init_dim = None
# dim = 128
# dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
# print(dims)
# sys.exit(0)

# base unet for diffusion model
# unet1 = Unet(
#     dim = 128,
#     cond_dim = 512,
#     dim_mults = (1, 2, 2, 4),
#     # dim_mults = (1, 2, 4, 8),
#     num_resnet_blocks = 3,
#     layer_attns = (False, True, True, True),
#     layer_cross_attns = (False, True, True, True)
# )


# # super-resolution unet for diffusion model
# unet2 = Unet(
#     dim = 128,
#     cond_dim = 512,
#     # dim_mults = (1, 2, 4, 8),
#     dim_mults = (1, 2, 2, 2),
#     num_resnet_blocks = (2, 4, 8, 8),
#     layer_attns = (False, False, False, True),
#     layer_cross_attns = (False, False, False, True)
# )

# # unets for unconditional imagen
# unet1 = Unet(
#     dim = 32,
#     dim_mults = (1, 2, 4),
#     num_resnet_blocks = 3,
#     layer_attns = (False, True, True),
#     layer_cross_attns = False,
#     use_linear_attn = True
# )

# # super-resolution unet for diffusion model
# unet2 = SRUnet256(
#     dim = 32,
#     dim_mults = (1, 2, 4),
#     num_resnet_blocks = (2, 4, 8),
#     layer_attns = (False, False, True),
#     layer_cross_attns = False
# )

# # imagen, which contains the unets above (base unet and super resoluting ones)
# imagen = Imagen(
#     condition_on_text = False,   # this must be set to False for unconditional Imagen
#     unets = (unet1, unet2),
#     image_sizes = (64, 128),
#     timesteps = 1000
# )


# this takes case of EMA for the unets
# I will likely need this
# trainer = ImagenTrainer(imagen).cuda()


# imagen, which contains the unets above (base unet and super resoluting ones)
# model = Imagen(
#     unets = (unet1, unet2),
#     image_sizes = (128, 256),
#     timesteps = 1000,
#     cond_drop_prob = 0.1
# ).cuda()

# # print(model)
# print(sum(p.numel() for p in model.parameters()) / 1e6)