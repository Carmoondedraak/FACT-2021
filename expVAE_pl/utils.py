import torch.tensor as tensor
import numpy as np
import cv2
from urllib import request
import tarfile
import torch
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
    ftpstream = request.urlopen(url)
    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
    thetarfile.extractall(path)

# TODO: Second inference method
def calc_latent_mu_var(model, dm, batch_size):
    stats = torch.zeros((dm.train_set.__len__(), 2))
    norm_mu, norm_var = [], []
    for i, (x, _) in enumerate(dm.train_dataloader()):
        mu, log_var = model.encode(x)
        norm_mu.append(mu)
        norm_var.append(log_var)

    norm_mu = torch.cat(norm_mu, dim=0).mean(0)
    norm_var = torch.cat(norm_var, dim=0).mean(0)

    return norm_mu, norm_var