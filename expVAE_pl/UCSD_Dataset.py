import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as trforms
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image
from utils import download_and_extract

# Custom PyTorch dataset for UCSD pedestrian dataset
class UCSD(data.Dataset):
    def __init__(self, path, transforms=None, train=True):
        self.train = train


        if self.train:
            self.path = os.path.join(path, 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train')
        else:
            self.path = os.path.join(path, 'UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test')

        if self.train:
            folders = sorted([os.path.join(self.path, folder) for folder in os.listdir(self.path) if 'Train' in folder])
            self.files = sorted([os.path.join(folder, file) for folder in folders for file in os.listdir(folder) if '.tif' in file])
        else:
            # folders = sorted([os.path.join(self.path, folder) for folder in os.listdir(self.path) if 'Test' in folder and '_gt' not in folder])
            # self.files = sorted([os.path.join(folder, file) for folder in folders for file in os.listdir(folder) if '.tif' in file])
            target_folders = sorted([os.path.join(self.path, folder) for folder in os.listdir(self.path) if 'Test' in folder and '_gt' in folder])
            folders = [folder.rstrip('_gt') for folder in target_folders]
            self.target_files = sorted([os.path.join(folder, file) for folder in target_folders for file in os.listdir(folder) if '.bmp' in file])
            self.files = sorted([os.path.join(folder, file) for folder in folders for file in os.listdir(folder) if '.tif' in file])

        self.transforms = transforms
        self.target_transforms = transforms

    def __getitem__(self, i):
        file_name = self.files[i]
        img = Image.open(file_name)
        if self.train:
            img = self.transforms(img)
            return img, torch.tensor(0)
        else:
            img = self.target_transforms(img)
            target_img = self.target_transforms(Image.open(self.target_files[i]))
            return img, target_img

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

        self.url = 'http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz'

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            download_and_extract(self.url, self.data_dir)

    def setup(self):
        self.train_set = UCSD(path=self.data_dir, transforms=self.transform, train=True)
        self.test_set = UCSD(path=self.data_dir, transforms=self.transform, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

