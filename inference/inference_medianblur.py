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
from ensamble import Ensamble
from utils.diffusion_restore import Diffusion
from defaultclassifier import DefaultClassifier
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
    parser.add_argument('-s',
                        '--step',
                        type=int,
                        required=False,
                        default=0,
                        help='steps')
    args = parser.parse_args()
    return args.config,args.step


cf_fd,step = parse_args()


def main():
    cmd_input = cf_fd
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
#diffusion = Diffusion(restore_timesteps=config['diffusion']['restore_timesteps'],
#                      timesteps=config['diffusion']['timesteps'],
#                      beta_schedule=config['diffusion']['beta_schedule'],
#                      image_size=config['data']['image_size'])

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
                          shuffle=False,
                          num_workers=5,
                          prefetch_factor=3)

# #### GET MODELS ##############################################################i
"""
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
"""
#model.load_state_dict(torch.load(config['inference']['model_ckpt'],map_location='cpu'))
#model.eval()

#print(sum(p.numel() for p in model.parameters()) / 1e6)
#model = model.cuda()

# Distribute the models to the device
#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model)
#    print(f"Using {torch.cuda.device_count()} GPUs.")
# summary(model, input_size=(config['model']['in_channels'],
#                         config['data']['image_size'],
#                         config['data']['image_size']))

##### GET OPTIMIZER ############################################################
#optimizer = Adam(model.parameters(), lr=config['train']['lr'])

# #### TRAIN MODELS ############################################################
# mark the time for saving results and models
now = datetime.datetime.now()
now = f'{str(now.day)}-{str(now.month)}-{str(now.year)}'
from tqdm import tqdm
"""
for idx,batch in enumerate(tqdm(train_loader)):
    img = batch['image'][0]
    img = (img + 1) / 2  # get out of batch and rescale
    sav = preprocess_numpy(img.numpy())
    if idx == 3000:
        break
    torch.save(sav,'LQpt/sample-'+str(idx)+'.pt')
"""
import cv2
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

#classifier = torch.load(config['inference']['classifier']).cuda()
for idx,batch in tqdm(enumerate(train_loader)):
    cc = batch['image'][0].numpy()
    cc = (cc+1)/2
    med = np.zeros(cc.shape)
    for i in range(2):
        med[i] = median_filter(cc[i],5)
    results_folder = Path('rawunpaired/'+config['inference']['result_path'])
    results_folder.mkdir(exist_ok=True)
    torch.save(med,results_folder/f'sample-{idx}.pt')
    medo = preprocess_numpy(med)
    # save images
    results_folder = Path('hequnpaired/'+config['inference']['result_path'])
    results_folder.mkdir(exist_ok=True)
    torch.save(medo,results_folder/f'sample-{idx}.pt')
