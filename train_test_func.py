
# import seaborn as sns

import torch
from torch import nn
import torch.optim as optim

from resnet import ResNet
from train import train

from torchvision import transforms
from DatasetPicker import DatasetPicker

from data_loader import get_data_loaders

import warnings

import os

warnings.filterwarnings("ignore")

# GLOBALS
# -----------------------


# SET FINAL TRANSFORMS WITH NORMALISATION

# [x] simple data augmentation in [24]
# [x] 4 pixels are padded on each side,
# [x] and a 32×32 crop is randomly sampled from the padded image or its horizontal flip.
# [x] For testing, we only evaluate the single view of the original 32×32 image.


def train_net(
    n=3,
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    milestones=[82, 123],
    gamma=0.1,
    plain=False,
    train_dataset=DatasetPicker.FASHION_MNIST,
    test_dataset=DatasetPicker.FASHION_MNIST
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

    data_dir = 'data/cifar10'
    batch_size = 128




    # Normalisation parameters fo CIFAR10
    means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
    stds  = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]

    normalize = transforms.Normalize(
        mean=means,
        std=stds,
    )

    train_transform = transforms.Compose([ 
        # 4 pixels are padded on each side, 
        transforms.Grayscale(num_output_channels=3),
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
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize
    ])

    if train_dataset == test_dataset:
        print("Getting data loaders for same dataset")
        train_loader, test_loader = get_data_loaders(
            data_dir,
            batch_size,
            train_transform,
            test_transform,
            train_dataset,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    else:
        print("Getting train data loader")
        train_loader, _ = get_data_loaders(
            data_dir,
            batch_size,
            train_transform,
            test_transform,
            train_dataset,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        print("Getting test data loader")
        _, test_loader = get_data_loaders(
            data_dir,
            batch_size,
            train_transform,
            test_transform,
            test_dataset,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    
    print(f"Training ResNet with {n} Layers")
    # Reload data

    # Create model
    model_folder = f'{"plain_" if plain else ""}resnet_{6*n+2}_{train_dataset.name}_{test_dataset.name}'
    if not os.path.exists("pretrained/" + model_folder):
        os.makedirs("pretrained/" + model_folder)
    

    for i in [20, 50, 100]:
        if not os.path.exists("pretrained/" + model_folder + f"/_{i}_epoch"):
            os.makedirs("pretrained/" + model_folder + f"/_{i}_epoch")
        if not os.path.exists("pretrained/" + model_folder + f"/_{i}_epoch/images"):
            os.makedirs("pretrained/" + model_folder + f"/_{i}_epoch/images")

    
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

    

    

    # results_name = f'results/{"plain_" if plain else ""}resnet_{6*n+2}/{"plain_" if plain else ""}resnet_{6*n+2}'
    
    model_names = [
        "pretrained/" + model_folder + "/" + "_20_epoch" ,
        "pretrained/" + model_folder + "/" + "_50_epoch" ,
        "pretrained/" + model_folder + "/" + "_100_epoch" ,
    ]
    # model_name = "pretrained/" + model_folder + "/" + model_folder
    train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        MODEL_PATHS=model_names,
    )


def test_net(
    test_loader,
    n=3,
    plain=False
):
    # GLOBALS
    # -----------------------
    model_file = f'pretrained/{"plain_" if plain else ""}resnet_{6*n+2}.pt'

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
