import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from configs.data_configs import DATASETS

def get_labels(dataset_name):
    return DATASETS[dataset_name]['labels']

def get_data_loaders(dataset_name, 
                     data_dir, 
                     batch_size, 
                     train_transform, 
                     test_transform, 
                     shuffle=True, 
                     num_workers=4, 
                     pin_memory=False):
    """
    Utility function for loading and returning train and test
    multi-process iterators over the specified dataset.
    If using CUDA, set pin_memory to True.

    Params
    ------
    - dataset_name: name of the dataset (e.g., 'cifar10', 'fashionMNIST')
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - train_transform: pytorch transforms for the training set
    - test_transform: pytorch transofrms for the test set
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - test_loader:  test set iterator.
    """

    # Retrieve dataset configuration
    dataset_config = DATASETS[dataset_name]

    # Load the datasets
    train_dataset = dataset_config['dataset_class'](
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    test_dataset = dataset_config['dataset_class'](
        root=data_dir, train=False,
        download=True, transform=test_transform,
    )

    # Create loader objects
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return (train_loader, test_loader)


def plot_images(images, cls_true, cls_pred=None):
    """
    Plot images with labels.
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """

    # Retrieve label names from dataset configuration
    label_names = get_labels('cifar10')  # Update this to use dataset_name

    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# Example usage:
dataset_name = 'cifar10'
data_dir = DATASETS[dataset_name]['data_dir']
batch_size = 32
train_transform = DATASETS[dataset_name]['transform_train']
test_transform = DATASETS[dataset_name]['transform_test']

train_loader, test_loader = get_data_loaders(dataset_name, 
                                              data_dir, 
                                              batch_size, 
                                              train_transform, 
                                              test_transform)

print(get_labels('cifar10'))
print(get_labels('fashionMNIST'))
		
