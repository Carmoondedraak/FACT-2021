import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as trforms
import cv2
import os
from utils import *

# Extends torchvision MNISt dataset, overwriting the data to only have a single class
class OneClassMNIST(MNIST):
    def __init__(self, digit, root, transforms, split=False, train=False, download=False):
        super(OneClassMNIST, self).__init__(root, download=download)

        self.digit = digit
        digit_idxs = torch.nonzero(self.targets == digit).squeeze(1)
        if split:
            if train:
                digit_idxs = digit_idxs[:int(len(digit_idxs)*0.9)]
            else:
                digit_idxs = digit_idxs[int(len(digit_idxs)*0.9):]

        self.data = self.data[digit_idxs, :,:]
        self.targets = torch.full(digit_idxs.shape, fill_value=digit)

        self.transform = transforms

# PyTorch Lightning datamodule, which handles train/test sets, preprocessing and downloading the dataset
class OneClassMNISTDataModule(pl.LightningDataModule):

    def __init__(self, root='./MNIST_dataset', train_digit=1, test_digit=9, batch_size=64, num_workers=4):
        self.dims = (1,28,28)
        self.data_dir, self.train_digit, self.test_digit, self.batch_size, self.num_workers = root, \
            train_digit, test_digit, batch_size, num_workers
        self.transform = trforms.Compose([
            trforms.ToTensor(),
        ])

        self.url = 'www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz'

    def prepare_data(self):
        OneClassMNIST(digit=1, root=self.data_dir, download=True, transforms=None)

    def setup(self, stage=None):
        if stage == 'train' or stage is None:
            self.train_set = OneClassMNIST(digit=self.train_digit, root=self.data_dir, transforms=self.transform,
                                            split=True, train=True)
            self.val_set = OneClassMNIST(digit=self.train_digit, root=self.data_dir, transforms=self.transform,
                                            split=True, train=False)

        if stage == 'test' or stage is None:
            self.test_set = OneClassMNIST(digit=self.train_digit, root=self.data_dir, transforms=self.transform,
                                            split=False, train=False)
        

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)