import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

from models.resnet import ResNet
from datasets.data_loader import get_data_loaders, plot_images, get_labels
from utils.utils import calculate_normalisation_params
from train import train


import warnings
warnings.filterwarnings('ignore')

# GLOBALS
# -----------------------

data_dir = 'data/cifar10'
batch_size = 128


# SET FINAL TRANSFORMS WITH NORMALISATION

# [x] simple data augmentation in [24]
# [x] 4 pixels are padded on each side,
# [x] and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
# [x] For testing, we only evaluate the single view of the original 32×32 image.


# Normalisation parameters fo CIFAR10
means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
stds  = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]

normalize = transforms.Normalize(
    mean=means,
    std=stds,
)

train_transform = transforms.Compose([
    # 4 pixels are padded on each side,
    transforms.Pad(4),
    # a 32×32 crop is randomly sampled from the
    # padded image or its horizontal flip.
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    # For testing, we only evaluate the single
    # view of the original 32×32 image.
    transforms.ToTensor(),
    normalize
])

def train_net(n=3,
              epochs=164,
              lr=0.1,
              momentum=0.9,
              weight_decay=0.0001,
              milestones=[82, 123],
              gamma=0.1,
              plain=False
    ):


    # TRAINING PARAMETERS
    # -------------------------

    # Authors cite 64k iterations
    # 64000/391 = 164
    # epochs = 164

    # OPTIMISER PARAMETERS
    # lr = 0.1 # authors cite 0.1
    # momentum = 0.9
    # weight_decay = 0.0001

    # LEARNING RATE ADJUSTMENT
    # Reduce learning rate at iterations
    # 32k and 48k. Convert to epochs:
    # e.g. iterations / (n_observations/batch_size)
    # 32000/391 = 82
    # 48000/391 = 123
    # milestones = [82, 123]
    # Divide learning rate by 10 at each milestone
    # gamma = 0.1

    # TRAIN PLAIN NETS
    # -------------------------

    # Train a plain network with 18 layers
    # and no shortcuts
    ns = [n]

    for n in ns:
        print(f'Training plain ResNet with {n} Layers')
        # Reload data
        train_loader, test_loader = get_data_loaders(
            data_dir,
            batch_size,
            train_transform,
            test_transform,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Create model
        if plain:
            model = ResNet(n, shortcuts=False)
        else:
            model = ResNet(n, shortcuts=True)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=milestones, gamma=gamma)
        results_file = f'results/plain_resnet_{6*n+2}.csv'
        model_file = f'pretrained/plain_resnet_{6*n+2}.pt'
        train(model,
              epochs,
              train_loader,
              test_loader,
              criterion,
              optimizer,
              results_file,
              scheduler=scheduler,
              MODEL_PATH=model_file)

def test_net(n=3):
    # GLOBALS
    # -----------------------
    model_file = f'pretrained/plain_resnet_{6*n+2}.pt'
    classes = get_labels()




        # SET FINAL TRANSFORMS WITH NORMALISATION

    # [x] simple data augmentation in [24]
    # [x] 4 pixels are padded on each side,
    # [x] and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
    # [x] For testing, we only evaluate the single view of the original 32×32 image.

    _, test_data = get_data_loaders(
        data_dir,
        batch_size,
        train_transform,
        test_transform,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    data_iter = iter(test_data)
    images, labels = next(data_iter)

    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    net = ResNet()
    net.load_state_dict(torch.load(model_file, weights_only=True))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))


if __name__ == '__main__':
    # train_net()
    test_net(9)



