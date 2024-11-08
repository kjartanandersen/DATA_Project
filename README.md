# Impact of CNNs on Image Classification

This repository contains the code used for the project **"Impact of CNNs on Image Classification"** for the course T-809-DATA. The project focuses on studying the effects of Convolutional Neural Networks (CNNs), specifically the ResNet architecture, on image classification tasks using the CIFAR-10 and Fashion-MNIST datasets.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Model Implementation](#model-implementation)
- [Experiments](#experiments)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Authors](#authors)

## Introduction

Convolutional Neural Networks (CNNs) are fundamental in computer vision tasks, particularly for image classification. This project investigates the ResNet architecture, its implementation, and its performance on two benchmark datasets: CIFAR-10 and Fashion-MNIST.

**Objectives:**

- Study CNN basics and their impact on image classification.
- Implement and experiment with ResNet, evaluating its performance on both datasets.
- Identify challenges and analyze how ResNet addresses them.

## Datasets

### CIFAR-10

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Fashion-MNIST

The Fashion-MNIST dataset consists of 70,000 28x28 grayscale images of fashion products in 10 classes:

- T-shirt/top
- Trousers
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Model Implementation

The ResNet model is implemented using PyTorch. It accepts a network size parameter `n` that dictates the total number of layers in the ResNet, calculated as `6n+2`. This allows for creating various ResNet configurations (e.g., ResNet-20, ResNet-32, ResNet-44) by adjusting the `n` value.

**Model Components:**

- **Input Layer**
- **Residual Blocks:** Three stacks (Stack1, Stack2, Stack3) with increasing filter sizes and decreasing spatial dimensions.
- **Output Layer**

## Experiments

We experimented with varying the layer depth and the number of training epochs. Specifically, we compared models with 20, 32, and 44 layers, trained for 20, 50, and 100 epochs on both datasets.

**Training Setup:**

- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum.
- **Learning Rate:** Starts at 0.1, divided by 10 when reaching the milestone of 82 epochs.
- **Batch Size:** 128
- **Loss Function:** Cross-Entropy Loss

## Results

The results from our experiments, including training/test error graphs and confusion matrices, are documented in the [T_809_DATA___Project_Report.pdf](T_809_DATA___Project_Report.pdf) available in this repository.

**Key Findings:**

- While increasing the number of layers generally enhances accuracy, deeper models do not always yield better performance and may lead to overfitting, particularly with smaller datasets.
- The ResNet-32 model demonstrated better overall performance compared to the deeper ResNet-44, indicating that there is an optimal depth for balancing accuracy and training efficiency.
- ResNet's use of skip connections effectively mitigates the degradation problem found in very deep networks, supporting stable training and preventing vanishing gradients.
- Training stability was observed after the initial phase, particularly after the learning rate was reduced, highlighting the importance of learning rate scheduling.
- Differences between datasets were evident: Fashion-MNIST models exhibited higher initial accuracy and more consistent training, while CIFAR-10 models required more epochs for significant accuracy gains.

These findings emphasize the need for careful model depth selection and training strategy when working with convolutional neural networks for image classification tasks. Further research could focus on hyperparameter tuning or the application to more complex datasets to explore ResNet's behavior in larger-scale settings.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy
- matplotlib
- seaborn


## Usage

### Training the Model
Train the ResNet model using the provided function train_net in the train_test_func.py script.

**Parameters:**
- **n:** Controls number of layers where number of layers is: 6n+2. Defaults to 3
- **lr:** learning rate. Defaults to 0.1
- **momentum:** momentum factor. Defaults to 0.9
- **weight_decay:** weight decay. Defaults to 0.0001
- **milestones:** epoch milestones for when to apply learning rate decay. Defaults to [82,123]
- **gamma:** multiplicative factor of learning rate decay. Defaults to 0.1
- **plain:** set to **True** if a plain CNN is being used, set to **False** if ResNet model is being used. Default to False
- **train_dataset:** uses the DatasetPicker enumerator class to pick a dataset for training 
- **test_dataset:** uses the DatasetPicker enumerator class to pick a dataset for testing

### Train a ResNet-20 model on CIFAR-10
```bash
train_net(n=3, train_dataset=DatasetPicker.CIFAR10, test_dataset=DatasetPicker.CIFAR10)
```
### Train on Fashion-MNIST:
```bash
train_net(n=3, train_dataset=DatasetPicker.FASHION_MNIST, test_dataset=DatasetPicker.FASHION_MNIST)
```

### Evaluating the Model

After training the model, its performance via plots was done using the scripts available in the [`Plots-and-figures`](https://github.com/kjartanandersen/DATA_Project/tree/main/Plots-and-figures) directory.

**Generate Training and Test Error Plots:**

You can generate plots for training and test error over epochs using:

```bash
python Plots-and-figures/TestErrprPlots.py
```

***Generate Confusion Matrix:***
To create a confusion matrix for the model's predictions:
```bash
python Plots-and-figures/confusion.py
```
***Generate Comparisson plots:***
To create a the plots for the Comparisson criteria in the report:
```bash
python Plots-and-figures/Comparisson.py
```

## Project Structure

!!!TIL DÆMIS BREYTA ÞEGAR ER KOMIÐ LOKA STRUCTURE!!!
```bash
DATA_Project/
├── data/                                           # Data loading and preprocessing scripts
├── Plots-and-figures                               # Script to evaluate the model
    ├── Plots/                                      # Save location of plots
    ├── resnet_20_CIFAR10_CIFAR10/                  # data for plots Cifar-10, 20 layers
    ├── resnet_20_FASHION_MNIST_FASHION_MNIST/      # data for plots Cifar-10, 32 layers
    ├── resnet_32_CIFAR10_CIFAR10/                  # data for plots Cifar-10, 44 layers
    ├── resnet_32_FASHION_MNIST_FASHION_MNIST       # data for plots Fashion, 20 layers
    ├── resnet_44_CIFAR10_CIFAR10                   # data for plots Fashion, 32 layers
    ├── resnet_44_FASHION_MNIST_FASHION_MNIST       # data for plots Fashion, 44 layers
    ├── Comparisson.py                              # Plots comparisson
    ├── TestErrprPlots.py                           # Plots test error vs Epochs for all 
    └── confusion.py                                # Generates confusion matrix plots
├── pretrained/                                     # ResNet trained models
├── data_loader.py/                                 # Script for loading torch dataloaders
├── DatasetPicker.py/                               # Enumerator class for picking datasets
├── train.py                                        # Script to train the model
├── train_test_func.py                              # Script for setting up the training
├── main.py                                         # Example script
├── requirements.txt                                # List of required Python packages
├── T_809_DATA___Project_Report.pdf                 # Detailed project report
└── README.md                                       # Project README file
```

## Authors

- Elvar Þór Sævarsson ([elvars20@ru.is](mailto:elvars20@ru.is))
- Guðjón Hafsteinn Kristinsson ([gudjonk24@ru.is](mailto:gudjonk24@ru.is))
- Kjartan Már Andersen ([kjartan23@ru.is](mailto:kjartan23@ru.is))
