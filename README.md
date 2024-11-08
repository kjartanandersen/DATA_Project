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

Results from the experiments, including training/test error graphs and confusion matrices, are detailed in the [T_809_DATA___Project_Report.pdf](T_809_DATA___Project_Report.pdf) included in this repository.

**Key Findings:**

- Increasing the number of layers generally improves accuracy but may lead to overfitting if not properly regularized.
- The ResNet architecture effectively mitigates the degradation problem observed in very deep networks.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy
- matplotlib


## Usage

### Training the Model
Train the ResNet model using the provided scripts.

Train a ResNet model on CIFAR-10
```bash
SKRIFA HVERNIG TRAINAD CIFAR
```
Train on ResNet model on Fashion-MNIST:
```bash
SKRIFA HVERNIG TRAINAD FASHION
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
├── models/                                         # ResNet model implementations
├── results/                                        # Results from experiments
├── train.py                                        # Script to train the model
├── Plots-and-figures                               # Script to evaluate the model
    ├── Plots/                                      # Save location of plots
    ├── resnet_20_CIFAR10_CIFAR10/                  # data for plots Cifar-10, 20 layers
    ├── resnet_20_FASHION_MNIST_FASHION_MNIST/      # data for plots Cifar-10, 32 layers
    ├── resnet_32_CIFAR10_CIFAR10/                  # data for plots Cifar-10, 44 layers
    ├── resnet_32_FASHION_MNIST_FASHION_MNIST       # data for plots Fashion, 20 layers
    ├── resnet_44_CIFAR10_CIFAR10                   # data for plots Fashion, 32 layers
    ├── resnet_44_FASHION_MNIST_FASHION_MNIST       # data for plots Fashion, 44 layers
    ├── Comparisson.py                              # Plots comparisson
    ├── TestErrprPlots.py                           # Plots test error vs Epochs for all models
    └── confusion.py                                # Generates confusion matrix plots for all models
├── requirements.txt                  # List of required Python packages
├── T_809_DATA___Project_Report.pdf   # Detailed project report
└── README.md                         # Project README file
```

## Authors

- Elvar Þór Sævarsson ([elvars20@ru.is](mailto:elvars20@ru.is))
- Guðjón Hafsteinn Kristinsson ([gudjonk24@ru.is](mailto:gudjonk24@ru.is))
- Kjartan Már Andersen ([kjartan23@ru.is](mailto:kjartan23@ru.is))
