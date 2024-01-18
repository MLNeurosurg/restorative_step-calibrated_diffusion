#!/usr/bin/env python3
"""
General dataset for SRH dataset.
Third channel is only generated as a postprocessing step to save compute and memory.
"""

import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .data_utils import image_loader
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import cv2
class Diffusion_Dataset(Dataset):

    def __init__(
            self,
            data: List,
            img_root: str,
            image_transforms: Optional[transforms.Compose] = None,
            rotate = False,
            withlabel = False) -> Dict:

        self.data = data
        if withlabel:
            self.data = [d[0] for d in data]
            self.labels = [d[1] for d in data]
        self.withlabel = withlabel
        self.rotate = rotate
        self.img_root = img_root
        self.image_transforms = image_transforms
        if rotate:
            a = pd.read_pickle('/nfs/turbo/umms-tocho/data/pix/yiwei_paired_df_new.pkl')
            self.rots = list(a['rots'])
            self.names = list(a['good'])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            image_ID = self.data[idx]
            img_path = os.path.join(self.img_root, image_ID.strip('/'))
            image = image_loader(img_path)
            assert image.shape[0] == 300 and image.shape[1] == 300
        except:
            idx = random.randint(0, len(self.data) - 1)
            image_ID = self.data[idx]
            img_path = os.path.join(self.img_root, image_ID.strip('/'))
            image = image_loader(img_path)
            assert image.shape[0] == 300 and image.shape[1] == 300


        if self.rotate:
            indx = self.names.index(image_ID)
            rot = self.rots[indx]
            if idx != indx:
                print(str(idx)+' '+str(indx))
            if rot == 'bad':
                return None
            im2 = np.zeros_like(image)
            rt = rot[1]
            if len(rt) < 3:
                print('strange error')
                return None
            im2[:,:,0] = cv2.warpAffine(image[:,:,0], np.linalg.inv(rt)[:2], image[:,:,0].shape)
            im2[:,:,1] = cv2.warpAffine(image[:,:,1], np.linalg.inv(rt)[:2], image[:,:,1].shape)
            image = im2
        # convert to tensor
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        #print(image)
        sample = {}
        sample['image'] = image
        sample['imageID'] = image_ID
        if self.withlabel:
            sample['label'] = self.labels[idx]
        return sample



