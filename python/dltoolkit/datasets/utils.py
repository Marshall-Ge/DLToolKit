#!/usr/bin/env python3

import logging
import numpy as np
import random
import torch
from torch.utils.data.distributed import DistributedSampler



logger = logging.getLogger(__name__)




def as_binary_vector(labels, num_classes):
    """
    Construct binary label vector given a list of label indices.
    Args:
        labels (list): The input label list.
        num_classes (int): Number of classes of the label vector.
    Returns:
        labels (numpy array): the resulting binary vector.
    """
    label_arr = np.zeros((num_classes,))

    for lbl in set(labels):
        label_arr[lbl] = 1.0
    return label_arr


def aggregate_labels(label_list):
    """
    Join a list of label list.
    Args:
        labels (list): The input label list.
    Returns:
        labels (list): The joint list of all lists in input.
    """
    all_labels = []
    for labels in label_list:
        for l in labels:
            all_labels.append(l)
    return list(set(all_labels))


def tensor_normalize(tensor, mean, std, func=None):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    if func is not None:
        tensor = func(tensor)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def get_random_sampling_rate(long_cycle_sampling_rate, sampling_rate):
    """
    When multigrid training uses a fewer number of frames, we randomly
    increase the sampling rate so that some clips cover the original span.
    """
    if long_cycle_sampling_rate > 0:
        assert long_cycle_sampling_rate >= sampling_rate
        return random.randint(sampling_rate, long_cycle_sampling_rate)
    else:
        return sampling_rate


def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to ``True`` to have the data reshuffled
            at every epoch.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        sampler (Sampler): the created sampler.
    """
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    return sampler


def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None