import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as trforms
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import os
import numpy as np
from PIL import Image
from utils import download_and_extract
import random

class ToTensor(object):
    def __call__(self, image):
        img = F.to_tensor(image)
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        diff = (100*100) - ((img == 0).sum() + (img == 1).sum())
        return img.to(torch.int)

# Custom PyTorch dataset for UCSD pedestrian dataset
class UCSD(data.Dataset):
    def __init__(self, path, transforms=None, target_transforms=None, split_type='train'):
        self.split_type = split_type


        if self.split_type == 'train' or self.split_type == 'val':
            self.path = os.path.join(path, 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train')
        else:
            self.path = os.path.join(path, 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test')

        if self.split_type == 'train' or self.split_type == 'val':
            folders = sorted([os.path.join(self.path, folder) for folder in os.listdir(self.path) if 'Train' in folder])
            self.files = sorted([os.path.join(folder, file) for folder in folders for file in os.listdir(folder) if '.tif' in file])
            random.shuffle(self.files)
            if self.split_type == 'train':
                self.files = self.files[:int(0.9*len(self.files))]
            elif self.split_type == 'val':
                self.files = self.files[int(0.9*len(self.files)):]
        elif self.split_type == 'test':
            target_folders = sorted([os.path.join(self.path, folder) for folder in os.listdir(self.path) if 'Test' in folder and '_gt' in folder])
            folders = [folder.rstrip('_gt') for folder in target_folders]
            self.target_files = sorted([os.path.join(folder, file) for folder in target_folders for file in os.listdir(folder) if '.bmp' in file])
            self.files = sorted([os.path.join(folder, file) for folder in folders for file in os.listdir(folder) if '.tif' in file])

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, i):
        file_name = self.files[i]
        img = Image.open(file_name)
        if self.split_type == 'test':
            img = self.transforms(img)
            target_img = self.target_transforms(Image.open(self.target_files[i]))
            return img, target_img
        else:
            img = self.transforms(img)
            return img, torch.tensor(0)

    def __len__(self):
        return len(self.files)

# PyTorch Lightning datamodule, which handles train/test sets, preprocessing and downloading the dataset
class UCSDDataModule(pl.LightningDataModule):
    def __init__(self, root='./UCSD_dataset', batch_size=64, num_workers=4):
        self.dims = (1,100,100)
        self.data_dir, self.batch_size, self.num_workers = root, batch_size, num_workers
        self.transform = trforms.Compose([
            trforms.Resize(self.dims[1:]),
            trforms.ToTensor(),
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

