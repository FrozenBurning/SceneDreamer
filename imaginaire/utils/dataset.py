# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import importlib

import torch
import torch.distributed as dist

from imaginaire.utils.distributed import master_only_print as print


def _get_train_and_val_dataset_objects(cfg):
    r"""Return dataset objects for the training and validation sets.

    Args:
        cfg (obj): Global configuration file.

    Returns:
        (dict):
          - train_dataset (obj): PyTorch training dataset object.
          - val_dataset (obj): PyTorch validation dataset object.
    """
    dataset_module = importlib.import_module(cfg.data.type)
    train_dataset = dataset_module.Dataset(cfg, is_inference=False)
    if hasattr(cfg.data.val, 'type'):
        for key in ['type', 'input_types', 'input_image']:
            setattr(cfg.data, key, getattr(cfg.data.val, key))
        dataset_module = importlib.import_module(cfg.data.type)
    val_dataset = dataset_module.Dataset(cfg, is_inference=True)
    print('Train dataset length:', len(train_dataset))
    print('Val dataset length:', len(val_dataset))
    return train_dataset, val_dataset


def _get_data_loader(cfg, dataset, batch_size, not_distributed=False,
                     shuffle=True, drop_last=True, seed=0):
    r"""Return data loader .

    Args:
        cfg (obj): Global configuration file.
        dataset (obj): PyTorch dataset object.
        batch_size (int): Batch size.
        not_distributed (bool): Do not use distributed samplers.

    Return:
        (obj): Data loader.
    """
    not_distributed = not_distributed or not dist.is_initialized()
    if not_distributed:
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=seed)
    num_workers = getattr(cfg.data, 'num_workers', 8)
    persistent_workers = getattr(cfg.data, 'persistent_workers', False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and (sampler is None),
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    return data_loader


def get_train_and_val_dataloader(cfg, seed=0):
    r"""Return dataset objects for the training and validation sets.

    Args:
        cfg (obj): Global configuration file.

    Returns:
        (dict):
          - train_data_loader (obj): Train data loader.
          - val_data_loader (obj): Val data loader.
    """
    train_dataset, val_dataset = _get_train_and_val_dataset_objects(cfg)
    train_data_loader = _get_data_loader(cfg, train_dataset, cfg.data.train.batch_size, drop_last=True, seed=seed)
    not_distributed = getattr(cfg.data, 'val_data_loader_not_distributed', False)
    not_distributed = 'video' in cfg.data.type or not_distributed
    val_data_loader = _get_data_loader(
        cfg, val_dataset, cfg.data.val.batch_size, not_distributed,
        shuffle=False, drop_last=getattr(cfg.data.val, 'drop_last', False), seed=seed)
    return train_data_loader, val_data_loader


def _get_test_dataset_object(cfg):
    r"""Return dataset object for the test set

    Args:
        cfg (obj): Global configuration file.

    Returns:
        (obj): PyTorch dataset object.
    """
    dataset_module = importlib.import_module(cfg.test_data.type)
    test_dataset = dataset_module.Dataset(cfg, is_inference=True, is_test=True)
    return test_dataset


def get_test_dataloader(cfg):
    r"""Return dataset objects for testing

    Args:
        cfg (obj): Global configuration file.

    Returns:
        (obj): Val data loader. It may not contain the ground truth.
    """
    test_dataset = _get_test_dataset_object(cfg)
    not_distributed = getattr(
        cfg.test_data, 'val_data_loader_not_distributed', False)
    not_distributed = 'video' in cfg.test_data.type or not_distributed
    test_data_loader = _get_data_loader(
        cfg, test_dataset, cfg.test_data.test.batch_size, not_distributed,
        shuffle=False)
    return test_data_loader
