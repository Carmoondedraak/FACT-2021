import os
import torch
import torchvision.transforms.functional as F
from utils import set_work_directory, download_and_extract, UnNormalize
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms as trforms
from PIL import Image

# Converts masks to tensor
class ToTensor(object):
    def __call__(self, image):
        img = F.to_tensor(image)

        # Make sure images are truly binary
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        return img.to(torch.float)

class MVTEC(Dataset):
    def __init__(self, path, class_name='bottle', transform=None, target_transform=None, 
                test_transform=None, train=True, condition=None):
        """
        Initialize the MVTEC dataset
            path - Path to the root folder of the dataset
            class_name - name of the class of objects we want
            transform - What kind of transforms to use on the training set
            target_transform - What kind of transfroms to use on the target masks
            test_transform - What kind of tarnsforms to use on the test set
            train - Whether we want the train or val/test set
            condition - If not none, will only grab the images of a certain condition such as 'good'
        """
        self.train, self.class_name, self.condition = train, class_name, condition

        # Open the particular folder for the class specified by class_name
        class_folder = os.path.join(path, class_name)

        # If train, we grab all images in the folder called "train"
        if self.train:
            train_folder_path = os.path.join(class_folder, 'train')
            train_folders = [os.path.join(train_folder_path, folder) for folder in sorted(os.listdir(train_folder_path))]
            self.data = [(os.path.join(folder, file), folder.rsplit('/')[-1]) for folder in train_folders for file in sorted(os.listdir(folder))]
        # Otherwise, we open the 'test' folder and also grab the ground_truth images
        else:
            test_folder_path = os.path.join(class_folder, 'test')
            test_folders = [os.path.join(test_folder_path, folder) for folder in sorted(os.listdir(test_folder_path))]
            self.data = [(os.path.join(folder, file), folder.rsplit('/')[-1]) for folder in test_folders for file in sorted(os.listdir(folder))]

            ground_truth_path = os.path.join(class_folder, 'ground_truth')
            ground_truth_folders = [os.path.join(ground_truth_path, folder) for folder in sorted(os.listdir(ground_truth_path))]
            self.gt_data = [(os.path.join(folder, file), folder.rsplit('/')[-1]) for folder in ground_truth_folders for file in sorted(os.listdir(folder))]

            # If condition is not None, we want to grab only the images of a given condition
            if self.condition is not None:
                self.data = [(path, label) for (path, label) in self.data if label == self.condition]
            else:
                self.data = [(path, label) for (path, label) in self.data if 'good' not in label]

        # Define the transforms to be used glboally
        self.transforms = transform
        self.target_transforms = target_transform
        self.test_transforms = test_transform

    def __getitem__(self, i):
        # Read the input image and condition label
        filename, label = self.data[i]
        img = Image.open(filename)

        # Train set images have no target mask as they are all "good", so we return zeros target mask
        if self.train:
            img = self.transforms(img)
            gt_img = torch.zeros((1, *img.shape[1:]))
            return img, gt_img
        # Test images only have mask if their condition is other than good
        elif self.train == False:
            # Use test_transforms on test images
            img = self.test_transforms(img)
            # Create zero mask for "good" conditioned images, otherwise load the corresponding mask
            # with target_transforms
            if label == 'good':
                gt_img = torch.zeros((1, *img.shape[1:]))
            else:
                gt_filename, gt_label = self.gt_data[i]
                gt_img = Image.open(gt_filename)
                gt_img = self.target_transforms(gt_img)
            return img, gt_img

    def __len__(self):
        return len(self.data)

class MVTECDataModule(LightningDataModule):
    def __init__(self, root='./MVTEC_dataset', class_name='bottle', batch_size=16, num_workers=4):
        self.dims = (3, 256, 256)
        self.data_dir, self.batch_size, self.num_workers = root, batch_size, num_workers

        self.ch_mu, self.ch_std = (0.5,0.5,0.5), (0.5,0.5,0.5)

        self.unnormalize = UnNormalize(self.ch_mu, self.ch_std, n_channels=3)

        # Training transforms, which include augmentation and normalization
        self.transform = trforms.Compose([
            trforms.Resize(self.dims[1:]),
            trforms.RandomHorizontalFlip(),
            trforms.RandomRotation(90),
            trforms.ToTensor(),
            trforms.Normalize(self.ch_mu, self.ch_std)
        ])

        # Transforms used on the target/mask images
        self.target_transform = trforms.Compose([
            trforms.Resize(self.dims[1:]),
            ToTensor()
        ])

        # Transforms used on the eval/test set
        self.test_transform = trforms.Compose([
            trforms.Resize(self.dims[1:]),
            trforms.ToTensor(),
            trforms.Normalize(self.ch_mu, self.ch_std)
        ])

        # List all available class names for images
        self.class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

        # Check if class name selected exists
        if class_name in self.class_names:
            self.class_name = class_name
        else:
            raise ValueError(f'Expected one of {self.class_names} classes but was given {class_name}')
        
        # URL for downloading the dataset
        self.url = 'ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
        

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            download_and_extract(self.url, self.data_dir)

    def setup(self):
        """
        Creates 3 Datasets
            train_set - For loading "good" conditioned images from the train folder
            val_set - For loading "good" conditioned images from the test folder for validation/training (we don't create separate splits for train and test in our case)
            eval_set - For loading any conditioned image for VAE evaluation
        """
        self.train_set = MVTEC(self.data_dir, self.class_name, self.transform, self.target_transform, self.test_transform, train=True)
        self.val_set = MVTEC(self.data_dir, self.class_name, self.transform, self.target_transform, self.test_transform, train=False, condition='good')
        self.eval_set = MVTEC(self.data_dir, self.class_name, self.transform, self.target_transform, self.test_transform, train=False)

    # For training, we return one dataloader, of the class to be trained on
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    # NOTE: For our purposes, we have no need for a separate validation and test set, so they are the exac same
    # For validation, we return two dataloaders. One for the class being trained on, and another of the "anomaly" class 
    def val_dataloader(self):
        trained_digit_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)
        eval_digit_loader = DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return [trained_digit_loader, eval_digit_loader]

    # For testing, we return two dataloaders. One for the class being trained on, and another of the "anomaly" class 
    def test_dataloader(self):
        trained_digit_loader = DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)
        eval_digit_loader = DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return [trained_digit_loader, eval_digit_loader]

    def unnormalize_batch(self, images):
        return self.unnormalize(images)