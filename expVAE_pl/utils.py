import torch.tensor as tensor
import numpy as np
import cv2
from urllib import request
import tarfile
import torch
import os
from torch.utils.data import DataLoader

# Combines input image and attention map into final colormap
def get_cam(image, gcam):
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    h, w, d = image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=np.float) + \
        np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    return gcam

# Helper function to download and extract files given a url to a specified path
# Used by UCSD_Dataset.py to download and extract the dataset
def download_and_extract(url, path):
    print('Downloading datasets...')
    ftpstream = request.urlopen(url)
    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
    print('Extracting datasets...')
    thetarfile.extractall(path)

# Calculates the mu and log_var for a given model on the entire
# dataset. Used for the second inference method (eq 4) in the paper
def calc_latent_mu_var(model, dm, batch_size):
    stats = torch.zeros((dm.train_set.__len__(), 2))
    norm_mu, norm_var = [], []
    for i, (x, _) in enumerate(dm.train_dataloader()):
        mu, log_var = model.encode(x)
        norm_mu.append(mu.detach().cpu())
        norm_var.append(log_var.detach().cpu())

    norm_mu = torch.cat(norm_mu, dim=0).mean(0)
    norm_var = torch.cat(norm_var, dim=0).mean(0)

    return norm_mu.detach(), norm_var.detach()

# Sets working directory to project directory, so that all dataset paths
# will be relative to the project directory
def set_work_directory():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

# Finds checkpoint and hyperparameter paths for a given directory
def get_ckpt_path(log_dir, args):
    if args.model_version is None:
        raise ValueError("Must provide argument --model_version pointing to existing model to test when using --eval True")
    
    version_folder = os.path.join(log_dir, 'lightning_logs', f'version_{args.model_version}')

    if not os.path.exists(version_folder):
        raise ValueError(f"Must provide a version number that exists, but provided version number has no directory '{version_folder}'")
    
    checkpoint_folder = os.path.join(version_folder, 'checkpoints')

    checkpoint_files = os.listdir(checkpoint_folder)
    checkpoint_filenames = [file for file in checkpoint_files if '.ckpt' in file]

    if len(checkpoint_files) == 0 or len(checkpoint_filenames) == 0:
        raise ValueError(f"Checkpoints path '{checkpoint_folder}' contains no .ckpt files")

    if len(checkpoint_filenames) > 1:
        raise ValueError(f"More than one checkpoint found in '{checkpoint_folder}'")

    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_filenames[0])

    hparams_path = [file for file in os.listdir(version_folder) if '.yaml' in file][0]

    return checkpoint_path, os.path.join(version_folder, hparams_path)

# Unnormalizes a batch of images given a mu's and std for each channel
class UnNormalize(torch.nn.Module):
    def __init__(self, mean, std, n_channels):
        self.mean, self.std = torch.tensor(mean).view(n_channels,1,1), torch.tensor(std).view(n_channels,1,1)

    def __call__(self, imgs):
        
        un_tensor = imgs * self.std.to(imgs.device)
        un_tensor = un_tensor + self.mean.to(imgs.device)
        return un_tensor
