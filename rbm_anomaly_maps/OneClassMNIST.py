import torch
from torchvision.datasets import MNIST

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
