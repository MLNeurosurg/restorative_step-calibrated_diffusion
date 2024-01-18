#!/usr/bin/env python3

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from skimage.filters import gaussian
from skimage.transform import resize
from typing import List, Tuple

import random
import numpy as np
import time
import pandas
import matplotlib.pyplot as plt
import PIL.Image as Image

from tifffile import imread


#### IO helper functions #######################################################
def get_data(series_df: pandas.DataFrame,
             data_root_path: str,
             patch_types: List = ['tumor', 'normal']) -> List:

    data = []
    for _, row in series_df.iterrows():
        for patch_type in patch_types:
            patch_path = os.path.join(row.center, row.study,
                                      str(int(row.series)), 'data', 'patches',
                                      patch_type)

            # read in files the series
            try:
                files = os.listdir(os.path.join(data_root_path, patch_path))
            except FileNotFoundError:
                print(f"FileNotFoundError: {patch_path}")
                continue

            for file in files:
                try:
                    file_path = os.path.join(patch_path, file)
                    data.append(file_path)
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    continue
    print(data[0])
    return data


def image_loader(path):
    try:
        image = imread(path).astype(float)

    # handle IO errors during training
    except FileNotFoundError:
        time.sleep(5)
        image = imread(path).astype(float)

    # move to channels last format
    image = np.moveaxis(image, 0, -1)
    #print(image)
    image /= 2**16
    #print(image.shape)
    #print(image[22:278,22:278,:])
    return image


def train_validation_split(data: List, validation_cases: List) -> Tuple:
    """Function to split the data into training and validation cases
    based on validation_cases list."""
    val_data = []
    train_data = []
    for i in data:
        for val_case in validation_cases:
            if val_case in i:
                val_data.append(i)
                break
        else:
            train_data.append(i)

    assert len(train_data) + len(val_data) == len(data)
    for val in validation_cases:
        for i in train_data:
            if val in i:
                print("WARNING: VALIDATION CASES in TRAINING DATA!!!")
    return train_data, val_data


##### Image helper functions ###################################################
def get_third_channel(two_channel_image: np.ndarray,
                      recenter_channel: int = 5000,
                      channels_last: bool = True) -> np.ndarray:
    """Function that will perform elementwise subtraction of a two channel image
    (index1(CH3) - index0(CH2)) and concatenate the result to the first index. 
    recenter_channel will be added to the subtracted image."""

    if channels_last:
        img = np.zeros(
            (two_channel_image.shape[0], two_channel_image.shape[1], 3),
            dtype=float)
        CH2 = two_channel_image[:, :, 0]
        CH3 = two_channel_image[:, :, 1]

    else:
        img = np.zeros(
            (3, two_channel_image.shape[1], two_channel_image.shape[2]),
            dtype=float)
        CH2 = two_channel_image[0, :, :]
        CH3 = two_channel_image[1, :, :]

    # check image scale
    if two_channel_image.max() > 1:
        subtracted_channel = (CH3 - CH2) + recenter_channel
    else:
        subtracted_channel = (CH3 - CH2) + (recenter_channel / 2**16)
    subtracted_channel[subtracted_channel < 0] = 0

    if channels_last:
        img[:, :, 0] = subtracted_channel
        img[:, :, 1] = CH2
        img[:, :, 2] = CH3
    else:
        img[0, :, :] = subtracted_channel
        img[1, :, :] = CH2
        img[2, :, :] = CH3

    return img


def percentile_rescaling(array: np.ndarray,
                         percentile_clip: int = 3) -> np.ndarray:
    """Function that will rescale a one channel image based on a percentile clipping.
    NOTE: percentile clip applies to the UPPER percentile. The lower percentile is fixed 
    at 3 percentile to avoid overly dark images."""
    p_low, p_high = np.percentile(array, (3, 100 - percentile_clip))
    # p_low, p_high = np.percentile(array, (2, 100 - percentile_clip))
    array = array.clip(min=p_low, max=p_high)
    img = (array - p_low) / (p_high - p_low)
    return img


def rescale_channels(patch,
                     percentile_clip=3,
                     subtracted_channel_recenter=0.0):

    # rescale the CH2 and CH3 channels
    patch = patch.astype(np.float)
    CH2 = percentile_rescaling(patch[:, :, 1], percentile_clip)
    assert (CH2.any() >= 0 and CH2.any() <= 1)
    CH3 = percentile_rescaling(patch[:, :, 2], percentile_clip)
    assert (CH3.any() >= 0 and CH3.any() <= 1)

    # channel subtraction
    subtracted_array = np.subtract(CH3, CH2)
    subtracted_array = subtracted_array.clip(min=0, max=1)

    # concatentate the postprocessed images
    img = np.zeros((CH2.shape[0], CH2.shape[1], 3), dtype=np.float)
    img[:, :, 0] = subtracted_array + subtracted_channel_recenter
    img[:, :, 1] = CH2
    img[:, :, 2] = CH3

    return img


def plot_images(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200, 200),
                            nrows=num_rows,
                            ncols=num_cols,
                            squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
# plot_images([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])

# fig = plt.figure()
# ims = []
# for i in range(timesteps):
#     im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
#     ims.append([im])

# animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# animate.save('diffusion.gif')
# plt.show()
