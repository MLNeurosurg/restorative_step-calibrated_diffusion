"""
# https://huggingface.co/blog/annotated-diffusion
# This tutorial is just for me to fully understand diffusion models.

https://github.com/lucidrains/denoising-diffusion-pytorch
"""
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
from data.data_utils import rescale_channels, get_third_channel
from torchvision.utils import save_image
from torchsummary import summary
from classifier import TClassifier
from pathlib import Path
from models.model import Unet
from models.model_utils import save_model

from data.data_utils import get_data, train_validation_split
from data.dataset import Diffusion_Dataset
from data.transforms import preprocess_transforms,preprocess_numpy
# from utils.diffusion import Diffusion
from utils.diffusion_restore import Diffusion

from runner import diffusion_runner
import numpy as np
torch.random.manual_seed(23)
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
                      image_size=config['data']['image_size'])

# #### GET DATALOADERS #########################################################
# train dataloader
train_dataset = Diffusion_Dataset(
    data=train_data,
    img_root=config['data']['data_root'],
    image_transforms=preprocess_transforms(
        image_size=config['data']['image_size'],centercrop=True),
)
train_loader = DataLoader(train_dataset,
                          #batch_size=config['train']['batch_size'],
                          batch_size=1,
                          shuffle=True,
                          num_workers=5,
                          pin_memory=True,
                          prefetch_factor=3)

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
                convnext_mult=config['model']['convnext_mult']).float()
if config['model']['model_type'] == 'imagen':
    pass

model.load_state_dict(torch.load(config['inference']['model_ckpt']))
model.eval()

print(sum(p.numel() for p in model.parameters()) / 1e6)
model = model.cuda()

# Distribute the models to the device
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs.")
# summary(model, input_size=(config['model']['in_channels'],
#                         config['data']['image_size'],
#                         config['data']['image_size']))

##### GET OPTIMIZER ############################################################
optimizer = Adam(model.parameters(), lr=config['train']['lr'])

# #### TRAIN MODELS ############################################################
# mark the time for saving results and models
now = datetime.datetime.now()
now = f'{str(now.day)}-{str(now.month)}-{str(now.year)}'
classifier = torch.load(config['inference']['classifier']).cuda()
for idx,batch in enumerate(train_loader):
    inimg = torch.zeros(1,3,256,256).cuda()
    inimg[:,1:3,:,:] = batch['image'].cuda()
    begin = classifier(inimg).long().item()
    if begin < config['inference']['skip_threshold']:
        print('skipped '+str(idx)+' because '+str(begin))
        continue
    outs1 = diffusion.sample(model,batch['image'].float().cuda(),1,begin=begin+1)
    outs11 = diffusion.sample(model,batch['image'].float().cuda(),1,begin=begin+1+config['inference']['additional_steps_1'])
    outs21 = diffusion.sample(model,batch['image'].float().cuda(),1,begin=begin+1+config['inference']['additional_steps_2'])
    outs = [outs1[0],outs1[-1],outs11[-1],outs21[-1]]
    rescaled_imgs = []
    for i, img in enumerate(outs):
        if True:
            img = (img[0] + 1) / 2  # get out of batch and rescale
            """
            img = get_third_channel(img, channels_last=False)
            img = np.swapaxes(img, 0, -1)
            img = rescale_channels(img)
            img = np.swapaxes(img, -1, 0)
            rescaled_imgs.append(
                torch.tensor(np.expand_dims(img, 0)))
            """
            rescaled_imgs.append(
                preprocess_numpy(img).unsqueeze(0))
    all_images = torch.cat(rescaled_imgs, dim=0)

    # save images
    results_folder = Path(config['inference']['result_path'])
    results_folder.mkdir(exist_ok=True)
    save_image(all_images,
               results_folder / f'sample-{idx}-{begin}.png',
               nrow=10)
