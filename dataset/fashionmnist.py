import numpy as np
from PIL import Image

import torchvision
import torch
from utils import utils


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def get_fashionmnist(root, n_labeled, transform_train=None, download=False):
    base_dataset = torchvision.datasets.FashionMNIST(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs = split(base_dataset.targets, int(n_labeled / 10))

    train_labeled_dataset = FashionMNIST_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = FashionMNIST_unlabeled(root, train_unlabeled_idxs, train=True,
                                                     transform=TransformTwice(transform_train))
    test_dataloader = utils.get_dataloader("fashionmnist", "test")

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataloader


def split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


class FashionMNIST_labeled(torchvision.datasets.MNIST):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(FashionMNIST_labeled, self).__init__(root, train=train,
                                                   transform=transform, target_transform=target_transform,
                                                   download=True)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = self.data.unsqueeze_(1) / 255.
        self.targets = torch.from_numpy(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FashionMNIST_unlabeled(FashionMNIST_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(FashionMNIST_unlabeled, self).__init__(root, indexs, train=train,
                                                     transform=transform, target_transform=target_transform,
                                                     download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
