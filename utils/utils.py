#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author：fmy
import torch
from torchvision import transforms, datasets



def get_dataloader(dataset_name, mode):
    torch.random.manual_seed(1.)
    sampler = torch.utils.data.SubsetRandomSampler(torch.randperm(10000)[:1000])
    if dataset_name in ["cifar10", "cifar100"]:
        if mode == "train":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif mode == "test":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        if dataset_name == "cifar10":
            dataset = datasets.CIFAR10('..//data', train=(mode == "train"),
                                       transform=transform)
        elif dataset_name == "cifar100":
            dataset = datasets.CIFAR100('..//data', train=(mode == "train"),
                                        transform=transform)

    elif dataset_name in ["mnist", "fashionmnist"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if dataset_name == "mnist":
            dataset = datasets.MNIST('..//data', train=(mode == "train"),
                                     transform=transform)
        if dataset_name == "fashionmnist":
            dataset = datasets.FashionMNIST('..//data', train=(mode == "train"),
                                            transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=256, sampler = sampler)
    return loader
