import os
import torch
from utils import set_work_directory
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms as trforms
from PIL import Image

class ToTensor(object):
    def __call__(self, image):
        img = F.to_tensor(image)
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        diff = (100*100) - ((img == 0).sum() + (img == 1).sum())
        return img.to(torch.int)

class MVTEC(Dataset):
    def __init__(self, path, class_name='bottle', transforms=None, target_transform=None,
                 train=True, condition=None):
        self.train, self.class_name = train, class_name

        class_folder = os.path.join(path, class_name)

        if self.train:
            train_folder_path = os.path.join(class_folder, 'train')
            train_folders = [os.path.join(train_folder_path, folder) for folder in sorted(os.listdir(train_folder_path))]
            self.data = [(os.path.join(folder, file), folder.rsplit('/')[-1]) for folder in train_folders for file in sorted(os.listdir(folder))]
        else:
            test_folder_path = os.path.join(class_folder, 'test')
            test_folders = [os.path.join(test_folder_path, folder) for folder in sorted(os.listdir(test_folder_path))]
            self.data = [(os.path.join(folder, file), folder.rsplit('/')[-1]) for folder in test_folders for file in sorted(os.listdir(folder))]


            ground_truth_path = os.path.join(class_folder, 'ground_truth')
            ground_truth_folders = [os.path.join(ground_truth_path, folder) for folder in sorted(os.listdir(ground_truth_path))]
            self.gt_data = [(os.path.join(folder, file), folder.rsplit('/')[-1]) for folder in ground_truth_folders for file in sorted(os.listdir(folder))]

        if condition is not None:
            self.data = [(path, label) for (path, label) in self.data if label == condition]

        self.transforms = transforms
        self.target_transforms = target_transform

    def __getitem__(self, i):
        filename, label = self.data[i]
        img = Image.open(filename)
        img = self.transforms(img)

        if self.train or self.condition is None:
            return img, torch.tensor(0)
        elif self.train == False:
            if label == 'good':
                gt_img = torch.zeros_like(img)
            else:
                gt_filename, gt_label = self.gt_data[i]
                gt_img = Image.open(gt_filename)
                gt_img = self.target_transforms(gt_img)

            return img, gt_img

    def __len__(self):
        return len(self.data)

class MVTECDataModule(LightningDataModule):
    def __init__(self, root='./MVTEC_dataset', class_name='bottle', batch_size=64, num_workers=4):
        # self.dims = (1, 256, 256)
        self.dims = (1, 100, 100)
        self.data_dir, self.batch_size, self.num_workers = root, batch_size, num_workers
        self.transform = trforms.Compose([
            trforms.Resize(self.dims[1:]),
            trforms.Grayscale(),
            trforms.ToTensor(),
        ])

        self.target_transform = trforms.Compose([
            trforms.Resize(self.dims[1:]),
            trforms.RandomHorizontalFlip()
            trforms.RandomRotation([10,25, 45, 90])
            trforms.ToTensor(),
        ])

        self.class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather'
                    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

        if class_name in self.class_names:
            self.class_name = class_name
        else:
            raise ValueError(f'Expected one of {self.class_names} classes but was given {class_name}')

        self.url = 'ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
        

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            download_and_extract(self.url, self.data_dir)

    def setup(self):
        self.train_set = MVTEC(self.data_dir, self.class_name, self.transform, self.target_transform, train=True)
        self.val_set = MVTEC(self.data_dir, self.class_name, self.transform, self.target_transform, train=False, condition='good')
        self.eval_set = MVTEC(self.data_dir, self.class_name, self.transform, self.target_transform, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        trained_digit_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        eval_digit_loader = DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return [trained_digit_loader, eval_digit_loader]

    def test_dataloader(self):
        trained_digit_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        eval_digit_loader = DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return [trained_digit_loader, eval_digit_loader]
