from torchvision.datasets import VisionDataset
import warnings
import torch
from PIL import Image
import os
import os.path
import numpy as np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageList(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, transform=None, target_transform=None):
        super(ImageList, self).__init__(root, transform=transform, target_transform=target_transform)

        # self.samples = np.loadtxt(root, dtype=np.unicode_, delimiter=' ')
        self.samples = np.loadtxt(root, dtype=np.dtype((np.unicode_, 1000)), delimiter=' ')
        self.loader = pil_loader

    def __getitem__(self, index):

        path, target = self.samples[index]
        target = int(target)

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path

    def __len__(self):
        return len(self.samples)

    def add_item(self, addition):
        self.samples = np.concatenate((self.samples, addition), axis=0)
        return self.samples

    def remove_item(self, reduced):
        self.samples = np.delete(self.samples, reduced, axis=0)
        return self.samples
