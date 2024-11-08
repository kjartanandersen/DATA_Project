import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the current directory where the .py file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, 'Plots', 'Confusion')

# Create the directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Function to load confusion matrices from files
def load_conf_matrix(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            # Clean up the line by removing non-numeric characters like brackets
            clean_line = line.strip().replace('[', '').replace(']', '')
            matrix.append(list(map(int, clean_line.split())))
    return np.array(matrix)

# File paths for CIFAR-10 (20, 32, and 44 layers at 20, 50, and 100 epochs)
file_path_20_20 = os.path.join(current_dir, 'resnet_20_CIFAR10_CIFAR10', '_20_epoch', '_conf_matrix.txt')
file_path_20_50 = os.path.join(current_dir, 'resnet_20_CIFAR10_CIFAR10', '_50_epoch', '_conf_matrix.txt')
file_path_20_100 = os.path.join(current_dir, 'resnet_20_CIFAR10_CIFAR10', '_100_epoch', '_conf_matrix.txt')

file_path_32_20 = os.path.join(current_dir, 'resnet_32_CIFAR10_CIFAR10', '_20_epoch', '_conf_matrix.txt')
file_path_32_50 = os.path.join(current_dir, 'resnet_32_CIFAR10_CIFAR10', '_50_epoch', '_conf_matrix.txt')
file_path_32_100 = os.path.join(current_dir, 'resnet_32_CIFAR10_CIFAR10', '_100_epoch', '_conf_matrix.txt')

file_path_44_20 = os.path.join(current_dir, 'resnet_44_CIFAR10_CIFAR10', '_20_epoch', '_conf_matrix.txt')
file_path_44_50 = os.path.join(current_dir, 'resnet_44_CIFAR10_CIFAR10', '_50_epoch', '_conf_matrix.txt')
file_path_44_100 = os.path.join(current_dir, 'resnet_44_CIFAR10_CIFAR10', '_100_epoch', '_conf_matrix.txt')

# File paths for Fashion-MNIST (20, 32, and 44 layers at 20, 50, and 100 epochs)
fashion_file_path_20_20 = os.path.join(current_dir, 'resnet_20_FASHION_MNIST_FASHION_MNIST', '_20_epoch', '_conf_matrix.txt')
fashion_file_path_20_50 = os.path.join(current_dir, 'resnet_20_FASHION_MNIST_FASHION_MNIST', '_50_epoch', '_conf_matrix.txt')
fashion_file_path_20_100 = os.path.join(current_dir, 'resnet_20_FASHION_MNIST_FASHION_MNIST', '_100_epoch', '_conf_matrix.txt')

fashion_file_path_32_20 = os.path.join(current_dir, 'resnet_32_FASHION_MNIST_FASHION_MNIST', '_20_epoch', '_conf_matrix.txt')
fashion_file_path_32_50 = os.path.join(current_dir, 'resnet_32_FASHION_MNIST_FASHION_MNIST', '_50_epoch', '_conf_matrix.txt')
fashion_file_path_32_100 = os.path.join(current_dir, 'resnet_32_FASHION_MNIST_FASHION_MNIST', '_100_epoch', '_conf_matrix.txt')

fashion_file_path_44_20 = os.path.join(current_dir, 'resnet_44_FASHION_MNIST_FASHION_MNIST', '_20_epoch', '_conf_matrix.txt')
fashion_file_path_44_50 = os.path.join(current_dir, 'resnet_44_FASHION_MNIST_FASHION_MNIST', '_50_epoch', '_conf_matrix.txt')
fashion_file_path_44_100 = os.path.join(current_dir, 'resnet_44_FASHION_MNIST_FASHION_MNIST', '_100_epoch', '_conf_matrix.txt')

# Load the confusion matrices for CIFAR-10
CIFAR_10_20_conf_matrix_20_epochs = load_conf_matrix(file_path_20_20)
CIFAR_10_20_conf_matrix_50_epochs = load_conf_matrix(file_path_20_50)
CIFAR_10_20_conf_matrix_100_epochs = load_conf_matrix(file_path_20_100)

CIFAR_10_32_conf_matrix_20_epochs = load_conf_matrix(file_path_32_20)
CIFAR_10_32_conf_matrix_50_epochs = load_conf_matrix(file_path_32_50)
CIFAR_10_32_conf_matrix_100_epochs = load_conf_matrix(file_path_32_100)

CIFAR_10_44_conf_matrix_20_epochs = load_conf_matrix(file_path_44_20)
CIFAR_10_44_conf_matrix_50_epochs = load_conf_matrix(file_path_44_50)
CIFAR_10_44_conf_matrix_100_epochs = load_conf_matrix(file_path_44_100)

# Load the confusion matrices for Fashion-MNIST
FASHION_20_conf_matrix_20_epochs = load_conf_matrix(fashion_file_path_20_20)
FASHION_20_conf_matrix_50_epochs = load_conf_matrix(fashion_file_path_20_50)
FASHION_20_conf_matrix_100_epochs = load_conf_matrix(fashion_file_path_20_100)

FASHION_32_conf_matrix_20_epochs = load_conf_matrix(fashion_file_path_32_20)
FASHION_32_conf_matrix_50_epochs = load_conf_matrix(fashion_file_path_32_50)
FASHION_32_conf_matrix_100_epochs = load_conf_matrix(fashion_file_path_32_100)

FASHION_44_conf_matrix_20_epochs = load_conf_matrix(fashion_file_path_44_20)
FASHION_44_conf_matrix_50_epochs = load_conf_matrix(fashion_file_path_44_50)
FASHION_44_conf_matrix_100_epochs = load_conf_matrix(fashion_file_path_44_100)

# Plotting function with save option
def plot_conf_matrix(conf_matrix, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path, bbox_inches='tight')  # Save the plot
    plt.show()

# Plot and save the confusion matrices for CIFAR-10
plot_conf_matrix(CIFAR_10_20_conf_matrix_20_epochs, 'Confusion Matrix for 20 Epochs (20 Layers, CIFAR-10)', os.path.join(plots_dir, '20_layer_Cifar_conf_matrix_20_epochs.png'))
plot_conf_matrix(CIFAR_10_20_conf_matrix_50_epochs, 'Confusion Matrix for 50 Epochs (20 Layers, CIFAR-10)', os.path.join(plots_dir, '20_layer_Cifar_conf_matrix_50_epochs.png'))
plot_conf_matrix(CIFAR_10_20_conf_matrix_100_epochs, 'Confusion Matrix for 100 Epochs (20 Layers, CIFAR-10)', os.path.join(plots_dir, '20_layer_Cifar_conf_matrix_100_epochs.png'))

plot_conf_matrix(CIFAR_10_32_conf_matrix_20_epochs, 'Confusion Matrix for 20 Epochs (32 Layers, CIFAR-10)', os.path.join(plots_dir, '32_layer_Cifar_conf_matrix_20_epochs.png'))
plot_conf_matrix(CIFAR_10_32_conf_matrix_50_epochs, 'Confusion Matrix for 50 Epochs (32 Layers, CIFAR-10)', os.path.join(plots_dir, '32_layer_Cifar_conf_matrix_50_epochs.png'))
plot_conf_matrix(CIFAR_10_32_conf_matrix_100_epochs, 'Confusion Matrix for 100 Epochs (32 Layers, CIFAR-10)', os.path.join(plots_dir, '32_layer_Cifar_conf_matrix_100_epochs.png'))

plot_conf_matrix(CIFAR_10_44_conf_matrix_20_epochs, 'Confusion Matrix for 20 Epochs (44 Layers, CIFAR-10)', os.path.join(plots_dir, '44_layer_Cifar_conf_matrix_20_epochs.png'))
plot_conf_matrix(CIFAR_10_44_conf_matrix_50_epochs, 'Confusion Matrix for 50 Epochs (44 Layers, CIFAR-10)', os.path.join(plots_dir, '44_layer_Cifar_conf_matrix_50_epochs.png'))
plot_conf_matrix(CIFAR_10_44_conf_matrix_100_epochs, 'Confusion Matrix for 100 Epochs (44 Layers, CIFAR-10)', os.path.join(plots_dir, '44_layer_Cifar_conf_matrix_100_epochs.png'))

## Plot and save the confusion matrices for Fashion-MNIST
plot_conf_matrix(FASHION_20_conf_matrix_20_epochs, 'Confusion Matrix for 20 Epochs (20 Layers, Fashion-MNIST)', os.path.join(plots_dir, '20_layer_FashionMNIST_conf_matrix_20_epochs.png'))
plot_conf_matrix(FASHION_20_conf_matrix_50_epochs, 'Confusion Matrix for 50 Epochs (20 Layers, Fashion-MNIST)', os.path.join(plots_dir, '20_layer_FashionMNIST_conf_matrix_50_epochs.png'))
plot_conf_matrix(FASHION_20_conf_matrix_100_epochs, 'Confusion Matrix for 100 Epochs (20 Layers, Fashion-MNIST)', os.path.join(plots_dir, '20_layer_FashionMNIST_conf_matrix_100_epochs.png'))

plot_conf_matrix(FASHION_32_conf_matrix_20_epochs, 'Confusion Matrix for 20 Epochs (32 Layers, Fashion-MNIST)', os.path.join(plots_dir, '32_layer_FashionMNIST_conf_matrix_20_epochs.png'))
plot_conf_matrix(FASHION_32_conf_matrix_50_epochs, 'Confusion Matrix for 50 Epochs (32 Layers, Fashion-MNIST)', os.path.join(plots_dir, '32_layer_FashionMNIST_conf_matrix_50_epochs.png'))
plot_conf_matrix(FASHION_32_conf_matrix_100_epochs, 'Confusion Matrix for 100 Epochs (32 Layers, Fashion-MNIST)', os.path.join(plots_dir, '32_layer_FashionMNIST_conf_matrix_100_epochs.png'))

plot_conf_matrix(FASHION_44_conf_matrix_20_epochs, 'Confusion Matrix for 20 Epochs (44 Layers, Fashion-MNIST)', os.path.join(plots_dir, '44_layer_FashionMNIST_conf_matrix_20_epochs.png'))
plot_conf_matrix(FASHION_44_conf_matrix_50_epochs, 'Confusion Matrix for 50 Epochs (44 Layers, Fashion-MNIST)', os.path.join(plots_dir, '44_layer_FashionMNIST_conf_matrix_50_epochs.png'))
plot_conf_matrix(FASHION_44_conf_matrix_100_epochs, 'Confusion Matrix for 100 Epochs (44 Layers, Fashion-MNIST)', os.path.join(plots_dir, '44_layer_FashionMNIST_conf_matrix_100_epochs.png'))
