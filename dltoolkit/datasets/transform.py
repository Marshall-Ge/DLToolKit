#!/usr/bin/env python3
# Copyright (c) 2023-2024. All rights reserved.

import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter


def get_train_transform(cfg):
    """
    Get transform for training mode with augmentations.

    Args:
        cfg (CfgNode): Configuration node from dltoolkit

    Returns:
        torchvision.transforms.Compose: Composed transforms for training
    """
    train_crop_size = cfg.DATA.TRAIN_CROP_SIZE if hasattr(cfg.DATA, 'TRAIN_CROP_SIZE') else 224
    input_mean = cfg.DATA.MEAN if hasattr(cfg.DATA, 'MEAN') else [0.485, 0.456, 0.406]
    input_std = cfg.DATA.STD if hasattr(cfg.DATA, 'STD') else [0.229, 0.224, 0.225]

    transform_list = [
        transforms.RandomResizedCrop(
            train_crop_size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0)
        ),
        transforms.RandomHorizontalFlip(),
    ]

    # Optional color jitter
    if hasattr(cfg.DATA, 'COLOR_JITTER') and cfg.DATA.COLOR_JITTER > 0:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.4 * cfg.DATA.COLOR_JITTER,
                contrast=0.4 * cfg.DATA.COLOR_JITTER,
                saturation=0.4 * cfg.DATA.COLOR_JITTER,
                hue=0.1 * cfg.DATA.COLOR_JITTER
            )
        )

    # Optional grayscale
    if hasattr(cfg.DATA, 'GRAYSCALE_PROB') and cfg.DATA.GRAYSCALE_PROB > 0:
        transform_list.append(transforms.RandomGrayscale(p=cfg.DATA.GRAYSCALE_PROB))

    # Optional gaussian blur
    if hasattr(cfg.DATA, 'GAUSSIAN_PROB') and cfg.DATA.GAUSSIAN_PROB > 0:
        transform_list.append(GaussianBlur(p=cfg.DATA.GAUSSIAN_PROB))

    # Optional PCA lighting jitter
    if hasattr(cfg.DATA, 'USE_PCA') and cfg.DATA.USE_PCA:
        transform_list.append(
            PCALighting(
                alphastd=0.1,
                eigval=cfg.DATA.TRAIN_PCA_EIGVAL,
                eigvec=cfg.DATA.TRAIN_PCA_EIGVEC
            )
        )

    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=input_mean,
            std=input_std
        )
    ])

    return transforms.Compose(transform_list)


def get_val_transform(cfg):
    """
    Get transform for validation mode.

    Args:
        cfg (CfgNode): Configuration node from dltoolkit

    Returns:
        torchvision.transforms.Compose: Composed transforms for validation
    """
    test_crop_size = cfg.DATA.TEST_CROP_SIZE if hasattr(cfg.DATA, 'TEST_CROP_SIZE') else 224
    input_mean = cfg.DATA.MEAN if hasattr(cfg.DATA, 'MEAN') else [0.485, 0.456, 0.406]
    input_std = cfg.DATA.STD if hasattr(cfg.DATA, 'STD') else [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize(int(test_crop_size / 0.875)),
        transforms.CenterCrop(test_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=input_mean,
            std=input_std
        )
    ])


def get_test_transform(cfg):
    """
    Get transform for test mode.

    Args:
        cfg (CfgNode): Configuration node from dltoolkit

    Returns:
        torchvision.transforms.Compose: Composed transforms for testing
    """
    return get_val_transform(cfg)


def build_transforms(cfg, mode="train"):
    """
    Factory function to build transforms.

    Args:
        cfg (CfgNode): Configuration node from dltoolkit
        mode (str): 'train', 'val', or 'test'

    Returns:
        torchvision.transforms.Compose: Composed transforms for specified mode
    """
    if mode == "train":
        return get_train_transform(cfg)
    elif mode == "val":
        return get_val_transform(cfg)
    else:
        return get_test_transform(cfg)


class GaussianBlur(object):
    """Apply gaussian blur with random kernel size and a probability p."""

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class PCALighting(object):
    """PCA-based lighting augmentation used in AlexNet."""

    def __init__(self, alphastd=0.1, eigval=None, eigvec=None):
        self.alphastd = alphastd
        self.eigval = eigval if eigval else [0.2175, 0.0188, 0.0045]
        self.eigvec = eigvec if eigvec else [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]

    def __call__(self, img):
        if self.alphastd == 0.0:
            return img

        img_tensor = torch.Tensor(np.array(img) / 255.0)
        if img_tensor.size(0) != 3:
            return img

        alpha = torch.normal(torch.zeros(3), self.alphastd)
        rgb = torch.sum(
            torch.stack([
                alpha[i] * torch.Tensor(self.eigvec[i]) * np.sqrt(self.eigval[i])
                for i in range(3)
            ]), dim=0
        )

        # Apply the lighting noise
        img_tensor = img_tensor + rgb.view(3, 1, 1)
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        # Convert back to PIL Image
        return transforms.ToPILImage()(img_tensor)


def get_simple_transform(size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Get a simple transform for basic use cases.

    Args:
        size (int): Image size after transformation
        mean (list): Normalization mean values
        std (list): Normalization standard deviation values

    Returns:
        torchvision.transforms.Compose: Simple composed transforms
    """
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_resize_transform(size):
    """
    Get a simple resize transform.

    Args:
        size (int or tuple): Target size

    Returns:
        torchvision.transforms.Resize: Resize transform
    """
    return transforms.Resize(size)


def get_tensor_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Get transform to convert PIL image to normalized tensor.

    Args:
        mean (list): Normalization mean values
        std (list): Normalization standard deviation values

    Returns:
        torchvision.transforms.Compose: Composed transforms for tensor conversion
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])