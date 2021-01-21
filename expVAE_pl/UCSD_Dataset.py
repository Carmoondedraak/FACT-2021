import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as trforms
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import os
import numpy as np
from PIL import Image
from utils import download_and_extract, UnNormalize
import random

# Converts masks to tensor
class ToTensor(object):
    def __call__(self, image):
        img = F.to_tensor(image)

        # Make sure images are truly binary
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        return img.to(torch.int)

# Custom PyTorch dataset for UCSD pedestrian dataset
class UCSD(data.Dataset):
    def __init__(self, path, transforms=None, target_transforms=None, split_type='train'):
        """
        Initialize the UCSD dataset
            path - Path to the root folder of the dataset
            transform - What kind of transforms to use on the training and validation set
            target_transform - What kind of transfroms to use on the target masks
            split_type - Either 'train', 'val' or 'test', where 'train' contains 90% of the
                data in the train folder, 'val' contains 10%, and 'test' contains all
                images in the test folder which also have a target mask provided
        """

        self.split_type = split_type

        # Train and val share the train folder, the test set will use all the data in the test set
        if self.split_type == 'train' or self.split_type == 'val':
            self.path = os.path.join(path, 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train')
        else:
            self.path = os.path.join(path, 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test')

        # The train folders are split 90% for training, 10% for validation
        if self.split_type == 'train' or self.split_type == 'val':
            # self.files contains the filenames of the images
            folders = sorted([os.path.join(self.path, folder) for folder in os.listdir(self.path) if 'Train' in folder])
            self.files = sorted([os.path.join(folder, file) for folder in folders for file in os.listdir(folder) if '.tif' in file])
            random.shuffle(self.files)
            if self.split_type == 'train':
                self.files = self.files[:int(0.9*len(self.files))]
            elif self.split_type == 'val':
                self.files = self.files[int(0.9*len(self.files)):]

        # From the test set, we only read from folders with target masks indicated by '_gt" in the folder names
        elif self.split_type == 'test':
            target_folders = sorted([os.path.join(self.path, folder) for folder in os.listdir(self.path) if 'Test' in folder and '_gt' in folder])
            folders = [folder.rstrip('_gt') for folder in target_folders]
            self.target_files = sorted([os.path.join(folder, file) for folder in target_folders for file in os.listdir(folder) if '.bmp' in file])
            self.files = sorted([os.path.join(folder, file) for folder in folders for file in os.listdir(folder) if '.tif' in file])

        # Define the transforms
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, i):
        # Read the input image
        file_name = self.files[i]
        img = Image.open(file_name)
        img = self.transforms(img)

        # If we're testing, we retrieve the target mask as well
        if self.split_type == 'test':
            target_img = self.target_transforms(Image.open(self.target_files[i]))
            return img, target_img
        else:
            return img, torch.zeros_like(img)

    def __len__(self):
        return len(self.files)

# PyTorch Lightning datamodule, which handles train/test sets, preprocessing and downloading the dataset
class UCSDDataModule(pl.LightningDataModule):
    def __init__(self, root='./UCSD_dataset', batch_size=64, num_workers=4):
        self.dims = (1,100,100)
        self.data_dir, self.batch_size, self.num_workers = root, batch_size, num_workers

        self.ch_mu, self.ch_std = 0.5, 0.5

        # self.unnormalize = UnNormalize(self.ch_mu, self.ch_std, n_channels=1)
        self.unnormalize = None

        self.transform = trforms.Compose([
            trforms.Resize(self.dims[1:]),
            trforms.ToTensor(),
            # trforms.Normalize(self.ch_mu, self.ch_std)
        ])

        self.target_transform = trforms.Compose([
            trforms.Resize(self.dims[1:]),
            ToTensor()
        ])

        self.url = 'http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz'

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            download_and_extract(self.url, self.data_dir)

    def setup(self):
        """
        Creates 3 Datasets
            train_set - For loading images containing only pedestrians for training
            val_set - For loading unseen images with only pedestrians
            eval_set - For loading unseen images with anomalies/vehicles amongst the pedestrians for evaluation
        """
        self.train_set = UCSD(path=self.data_dir, transforms=self.transform, target_transforms=self.target_transform, split_type='train')
        self.val_set = UCSD(path=self.data_dir, transforms=self.transform, target_transforms=self.target_transform, split_type='val')
        self.eval_set = UCSD(path=self.data_dir, transforms=self.transform, target_transforms=self.target_transform, split_type='test')

    # For training, we return one dataloader, of the class to be trained on
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    # NOTE: For our purposes, we have no need for a separate validation and test set, so they are the exac same
    # For validation, we return two dataloaders. One for the class being trained on, and another of the "anomaly" class 
    def val_dataloader(self):
        trained_digit_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        eval_digit_loader = DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return [trained_digit_loader, eval_digit_loader]

    # For testing, we return two dataloaders. One for the class being trained on, and another of the "anomaly" class 
    def test_dataloader(self):
        trained_digit_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        eval_digit_loader = DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return [trained_digit_loader, eval_digit_loader]

    def unnormalize_batch(self, images):
        if self.unnormalize is None:
            return images
        else:
            return self.unnormalize(images)