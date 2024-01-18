from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, Resize
from torchvision import transforms
import numpy as np

import torch

UINT16_MAX=65535.0
R_bias=5000

def get_third_channel(two_channel_image):
    """Helper function to generate our third channel from our two channel images."""
    img = np.zeros((3, two_channel_image.shape[1], two_channel_image.shape[2]), dtype=np.float32)

    CH2 = two_channel_image[0,:,:].astype(np.float32)
    CH3 = two_channel_image[1,:,:].astype(np.float32)
    if np.sum(CH3)>0:
        subtracted_channel = (CH3 - CH2) + R_bias
    else:
        subtracted_channel = CH3
    subtracted_channel = subtracted_channel.clip(0)
    img[0,:,:] = subtracted_channel
    img[1,:,:] = CH2
    img[2,:,:] = CH3
    return img

def histogram_equalization(im, bits):
    #https://ianwitham.wordpress.com/tag/histogram-equalization/
    #get image histogram
    num_bins = 2**bits
    imhist, bins = np.histogram(im.flatten(),num_bins)
    imhist[0] = 0

    cdf = imhist.cumsum() #cumulative distribution function
    cdf ** .5
    cdf = (2**bits-1) * cdf / cdf[-1] #normalize
    #cdf = cdf / (2**16.)  #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return np.array(im2, int).reshape(im.shape)


def preprocess_numpy(img, out_channels=3, histeq=True):
    if out_channels == 2:
        result = img.copy().astype(np.float32)
        if histeq:
            result[0,:,:] = histogram_equalization(img[0,:,:], 16)
            result[1,:,:] = histogram_equalization(img[1,:,:], 16)
    else:
        if histeq:
            img[0,:,:] = histogram_equalization(img[0,:,:], 16)
            img[1,:,:] = histogram_equalization(img[1,:,:], 16)
        result = get_third_channel(img)

    result = result / UINT16_MAX
    result = torch.from_numpy(result)
    result = torch.clamp(result, min=0., max=1.)
    return result



def preprocess_transforms(
    image_size: int = 256,
    random_crop_scale: tuple = (1.0, 1.0),
    centercrop=False
) -> transforms.Compose:
    if centercrop:
        print('we centercropped!')
        return Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda t: (t * 2) - 1)])
    preprocess_transform = Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    return preprocess_transform


# def reverse_transforms(image_size: int = 300) -> transforms.Compose:
#     reverse_transform = Compose([
#         Lambda(lambda t: (t + 1) / 2),
#         Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#         # Lambda(lambda t: t * 2**16),
#         # Lambda(lambda t: t.numpy().astype(float)),
#         # ToPILImage(),
#     ])
#     return reverse_transform


def reverse_transforms(image_size: int = 256) -> transforms.Compose:
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        # Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        # Lambda(lambda t: t * 2**16),
        # Lambda(lambda t: t.numpy().astype(float)),
        # ToPILImage(),
    ])
    return reverse_transform
