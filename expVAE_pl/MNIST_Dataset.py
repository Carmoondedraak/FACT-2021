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
    def __init__(self, digit, root, transforms, download=False, train=True):
        super(OneClassMNIST, self).__init__(root, train=train, download=download)

        # Get indices of images of specified digit
        self.digit = digit
        digit_idxs = torch.nonzero(self.targets == digit).squeeze(1)

        # Overwrite data and targets to only contain the images of the specified digit
        self.data = self.data[digit_idxs, :,:]
        self.targets = torch.full(digit_idxs.shape, fill_value=digit)

        # Overwrite the transforms to be used
        self.transform = transforms

# PyTorch Lightning datamodule, which handles train/test sets, preprocessing and downloading the dataset
class OneClassMNISTDataModule(pl.LightningDataModule):

    def __init__(self, root='./MNIST_dataset', train_digit=1, test_digit=9, batch_size=64, num_workers=4):
        # Define image size, globals and transforms
        self.dims = (1,28,28)
        self.data_dir, self.train_digit, self.test_digit, self.batch_size, self.num_workers = root, \
            train_digit, test_digit, batch_size, num_workers
        self.transform = trforms.Compose([
            trforms.ToTensor(),
        ])

        # URL used for downloading the dataset
        self.url = 'www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz'

    def prepare_data(self):
        """
        Download the dataset to data_dir if it hasn't been downloaded already
        """
        OneClassMNIST(digit=1, root=self.data_dir, download=True, transforms=None)

    def setup(self, stage=None):
        """
        Creates 3 Datasets
            train_set - For loading images of the specified digit to be trained on
            test_set - For loading images of the specified digit to be trained on, but from the test set
            eval_set - For loading images of other digits the model hasn't seen for evaluation
        """
        self.train_set = OneClassMNIST(digit=self.train_digit, root=self.data_dir, transforms=self.transform, train=True)
        self.test_set = OneClassMNIST(digit=self.train_digit, root=self.data_dir, transforms=self.transform, train=False)
        self.eval_set = OneClassMNIST(digit=self.test_digit, root=self.data_dir, transforms=self.transform)
        
    # For training, we return one dataloader, of the class to be trained on
    def train_dataloader(self):
        trained_digit_loader = DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return trained_digit_loader

    # NOTE: For our purposes, we have no need for a separate validation and test set, so they are the exac same
    # For validation, we return two dataloaders. One for the class being trained on, and another of the "anomaly" class 
    def val_dataloader(self):
        trained_digit_loader = DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        eval_digit_loader = DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return [trained_digit_loader, eval_digit_loader]

    # For testing, we return two dataloaders. One for the class being trained on, and another of the "anomaly" class 
    def test_dataloader(self):
        trained_digit_loader = DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        eval_digit_loader = DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return [trained_digit_loader, eval_digit_loader]