import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import copy
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from utils.losses import p_losses
from data.transforms import reverse_transforms
from data.data_utils import rescale_channels, get_third_channel
from collections import defaultdict


def _easy_image_viewer(images):
    img = images[-1, :, :, :].cpu().detach()
    img = img.swapaxes(0, -1)
    plt.imshow(img[:, :, 0])
    plt.show()


def diffusion_runner(dataloader,
                     model,
                     diffusion,
                     optimizer,
                     train=True,
                     epoch=0,
                     loss_type="l1",
                     restore_timesteps=100,
                     timesteps=200,
                     iterations=1000,
                     modulo_save=500,
                     modulo_diffusion_save=10,
                     save_path=None,
                     randomzero = False,
                     cond=False):

    device = next(model.parameters()).device
    if train:
        model.train()
    else:
        model.eval()

    # training loop
    losses = defaultdict(list)
    for idx, batch in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        batch_size = batch["image"].shape[0]
        channels = batch["image"].shape[1]
        images = batch["image"].to(device).float()
        # rev_batch = reverse_transforms()(images)
        # _easy_image_viewer(rev_batch)
        # continue

        # Sample t uniformally for every example in the batch
        assert timesteps >= restore_timesteps
        t = torch.randint(0, restore_timesteps, (batch_size, ), device=device).long()
        if cond:
            tt = torch.randint(0, timesteps, (batch_size,), device=device).long()
            x_noisy = diffusion.q_sample(x_start=images, t=t)
            #x_noisy = copy.deepcopy(images)
            loss = p_losses(model,diffusion, images, tt, loss_type = loss_type, randomzero=randomzero, cond=x_noisy)
        else:
            loss = p_losses(model, diffusion, images, t, loss_type=loss_type,randomzero=randomzero)
        noise = None
        if randomzero:
            loss, noise = loss
        losses["loss"].append(loss.cpu().item())
        losses["imageID"].append(batch["imageID"])

        if idx % 100 == 0:
            print("Loss:", loss.item())

        # update model
        if train:
            loss.backward()
            optimizer.step()

        # # save generated images
        # if not train and save_path is not None:
        if save_path is not None:

            save_interval = len(dataloader) // modulo_save
            if idx % save_interval == 0:
                all_images_list = list(
                    map(
                        lambda n: diffusion.sample(
                            # model, batch_size=1, channels=channels), [1]))
                            model, images, batch_size=1, channels=channels, train=True,noise=noise), [1]))

                # postprocess image for saving/visualization
                rescaled_imgs = []
                for i, img in enumerate(all_images_list[0]):
                    if i % modulo_diffusion_save == 0 or i == len(all_images_list[0])-1:
                        img = (img[0] + 1) / 2  # get out of batch and rescale
                        #### TODO: where we need to change the the postprocessing steps
                        img = get_third_channel(img, channels_last=False)
                        img = np.swapaxes(img, 0, -1)
                        img = rescale_channels(img)
                        img = np.swapaxes(img, -1, 0)
                        ####
                        rescaled_imgs.append(
                            torch.tensor(np.expand_dims(img, 0)))
                all_images = torch.cat(rescaled_imgs, dim=0)

                # save images
                results_folder = Path(save_path)
                results_folder.mkdir(exist_ok=True)
                save_image(all_images,
                           results_folder / f'sample-{epoch}-{idx}.png',
                           nrow=10)

        # break at final iteration
        if train:
            if idx >= iterations:
                break

    return losses
