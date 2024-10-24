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

from resnet import ResNet
from data_loader import get_data_loaders, plot_images, get_labels
from utils import calculate_normalisation_params
from train import train


import warnings

warnings.filterwarnings("ignore")

# GLOBALS
# -----------------------


# SET FINAL TRANSFORMS WITH NORMALISATION

# [x] simple data augmentation in [24]
# [x] 4 pixels are padded on each side,
# [x] and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
# [x] For testing, we only evaluate the single view of the original 32×32 image.


def train_net(
    train_loader,
    test_loader,
    n=3,
    epochs=164,
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    milestones=[82, 123],
    gamma=0.1,
    plain=False,
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
        print(f"Training plain ResNet with {n} Layers")
        # Reload data

        # Create model
        if plain:
            model = ResNet(n, shortcuts=False)
        else:
            model = ResNet(n, shortcuts=True)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )

        results_file = f'results/{"plain_" if plain else ""}resnet_{6*n+2}.csv'
        model_file = f'pretrained/{"plain_" if plain else ""}resnet_{6*n+2}.pt'
        train(
            model,
            epochs,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            results_file,
            scheduler=scheduler,
            MODEL_PATH=model_file,
        )


def test_net(
    test_loader,
    train_transform,
    test_transform,
    n=3,
    plain=False,
    data_dir="data/cifar10",
    batch_size=128,
):
    # GLOBALS
    # -----------------------
    model_file = f'pretrained/{"plain_" if plain else ""}resnet_{6*n+2}.pt'
    classes = get_labels()

    # SET FINAL TRANSFORMS WITH NORMALISATION

    # [x] simple data augmentation in [24]
    # [x] 4 pixels are padded on each side,
    # [x] and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
    # [x] For testing, we only evaluate the single view of the original 32×32 image.

    if plain:
        net = ResNet(n, shortcuts=False)
    else:
        net = ResNet(n, shortcuts=True)
    net.load_state_dict(torch.load(model_file, weights_only=True))

    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")

    """
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

    if plain:
        net = ResNet(n, shortcuts=False)
    else:
        net = ResNet(n, shortcuts=True)
    net.load_state_dict(torch.load(model_file, weights_only=True))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

    """
