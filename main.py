
import sys
from typing import TextIO
import json
import yaml
import argparse
import datetime
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image
from torchsummary import summary

from models.model import Unet
from models.model_utils import save_model

from data.data_utils import get_data, train_validation_split
from data.dataset import Diffusion_Dataset
from data.transforms import preprocess_transforms
# from utils.diffusion import Diffusion
from utils.diffusion_restore import Diffusion

from runner import diffusion_runner


#### OPTIONS ###################################################################
# get config file
def parse_args() -> TextIO:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


cf_fd = parse_args()


def main():
    cmd_input: TextIO = parse_args()
    if cmd_input.name.endswith(".json"):
        return json.load(cf_fd)
    elif cmd_input.name.endswith(".yaml") or cf_fd.name.endswith(".yml"):
        return yaml.load(cf_fd, Loader=yaml.FullLoader)


config = main()

#### DATA PREP #################################################################
if config['data']['from_patient_spreadsheet']:
    train_df = pd.read_excel(config['data']['data_spreadsheet'],
                             sheet_name=config['train']['sheetname'])
    image_data = get_data(train_df, config['data']['data_root'])
    train_data, val_data = train_validation_split(image_data,
                                                  config['val']['val_cases'])
else:
    image_data = pd.read_csv(
        config['data']['data_spreadsheet']).file_name.tolist()
    train_data, val_data = train_validation_split(image_data,
                                                  config['val']['val_cases'])
print(config['val']['val_cases'])
assert len(train_data) + len(val_data) == len(image_data), "Data split error"

#### INSTANTIATE DIFFUSION OBJECT ##############################################
diffusion = Diffusion(restore_timesteps=config['diffusion']['restore_timesteps'],
                      timesteps=config['diffusion']['timesteps'],
                      beta_schedule=config['diffusion']['beta_schedule'],
                      image_size=config['data']['image_size'],
                      cond=config['model']['cond'])

# #### GET DATALOADERS #########################################################
# train dataloader
train_dataset = Diffusion_Dataset(
    data=train_data,
    img_root=config['data']['data_root'],
    image_transforms=preprocess_transforms(
        image_size=config['data']['image_size']),
)
train_loader = DataLoader(train_dataset,
                          batch_size=config['train']['batch_size'],
                          shuffle=True,
                          num_workers=5,
                          pin_memory=True,
                          prefetch_factor=3)
print(len(train_loader))

# validation dataloader
val_dataset = Diffusion_Dataset(
    data=val_data,
    img_root=config['data']['data_root'],
    image_transforms=preprocess_transforms(
        image_size=config['data']['image_size']),
)
val_loader = DataLoader(val_dataset,
                        batch_size=config['train']['batch_size'],
                        shuffle=False,
                        num_workers=5,
                        pin_memory=True,
                        prefetch_factor=3)
print(len(val_loader))

# #### GET MODELS ##############################################################
if config['model']['model_type'] == 'unet':
    model = Unet(dim=config['data']['image_size'],
                channels=config['model']['in_channels'],
                init_dim=config['model']['init_dim'],
                out_dim=config['model']['out_dim'],
                dim_mults=config['model']['dim_mults'],
                with_time_emb=config['model']['with_time_emb'],
                resnet_block_groups=config['model']['resnet_block_groups'],
                use_convnext=config['model']['use_convnext'],
                convnext_mult=config['model']['convnext_mult'],
                cond=config['model']['cond']).float()
if config['model']['model_type'] == 'imagen':
    pass

if 'ckpt' in config['model']:
    sd = torch.load(config['model']['ckpt'])
    model.load_state_dict(sd)
    print('State dict loaded!')

print(sum(p.numel() for p in model.parameters()) / 1e6)
model = model.to('cuda:0')

# summary(model, input_size=(config['model']['in_channels'],
#                         config['data']['image_size'],
#                         config['data']['image_size']))

##### GET OPTIMIZER ############################################################
optimizer = Adam(model.parameters(), lr=config['train']['lr'])

# #### TRAIN MODELS ############################################################
# mark the time for saving results and models
now = datetime.datetime.now()
now = f'{str(now.day)}-{str(now.month)}-{str(now.year)}'

# initialize dictionaries to store results
for epoch in range(1, config['train']['n_epochs'] + 1):
    print(f'======================== {epoch} =========================')
    #"""
    # TRAINING
    train_losses = diffusion_runner(train_loader,
                                    model,
                                    diffusion,
                                    optimizer,
                                    train=True,
                                    epoch=epoch,
                                    loss_type=config['train']['loss_type'],
                                    restore_timesteps=config['diffusion']['restore_timesteps'],
                                    timesteps=config['diffusion']['timesteps'],
                                    iterations=config['train']['iters'],
                                    randomzero=config['diffusion']['randomzero'],
                                    cond=model.cond)
    #"""
    with torch.no_grad():
    # VALIDATION
        val_losses = diffusion_runner(val_loader,
                                      model,
                                      diffusion,
                                      optimizer,
                                      train=False,
                                      epoch=epoch,
                                      loss_type=config['train']['loss_type'],
                                      restore_timesteps=config['diffusion']['restore_timesteps'],
                                      timesteps=config['diffusion']['timesteps'],
                                      modulo_save=config['val']['modulo_save'],
                                      save_path=config['data']['image_save_path'],
                                      randomzero=config['diffusion']['randomzero'],
                                      cond=model.cond)

    save_model(
        model,
        f'{config["model"]["model_save_path"]}/pretrain_{epoch}_{now}_model')

    # save the model
    torch.save(
        train_losses,
        f'{config["train"]["metrics_save_path"]}/pretrain_{epoch}_{now}_metrics.pt'
    )
    torch.save(
        val_losses,
        f'{config["val"]["metrics_save_path"]}/pretrain_{epoch}_{now}_metrics.pt'
    )
