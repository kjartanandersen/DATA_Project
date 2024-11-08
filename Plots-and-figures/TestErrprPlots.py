import matplotlib.pyplot as plt
import numpy as np
import os

# Get the current directory and create the Plots folder if it doesn't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plots_dir, exist_ok=True)

# Epochs and layer depths
epochs = [20, 50, 100]
layer_depths = [20, 32, 44]

# Training times and accuracies for CIFAR-10 (real data from LaTeX table)
training_times_cifar10 = {
    20: [838, 2074, 4113],  # 20 layers
    32: [973, 2429, 4851],  # 32 layers
    44: [984, 2452, 4855]   # 44 layers
}
accuracies_cifar10 = {
    20: [100 - 19.33, 100 - 17.78, 100 - 10.04],  # 20 layers
    32: [100 - 19.85, 100 - 14, 100 - 9.36],      # 32 layers
    44: [100 - 18.56, 100 - 16.26, 100 - 9.66]    # 44 layers
}

# Training times and accuracies for Fashion-MNIST (real data from LaTeX table)
training_times_fashion = {
    20: [633, 1586, 3172],  # 20 layers
    32: [686, 1718, 3447],  # 32 layers
    44: [729, 1819, 3633]   # 44 layers
}
accuracies_fashion = {
    20: [100 - 8.92, 100 - 7.84, 100 - 5.18],     # 20 layers
    32: [100 - 7.87, 100 - 7.52, 100 - 5.17],     # 32 layers
    44: [100 - 9.04, 100 - 7.57, 100 - 5.69]      # 44 layers
}

# Define consistent colors for the layers
colors = {20: 'blue', 32: 'orange', 44: 'green'}

# Plot 1: Accuracy Across Epochs
plt.figure(figsize=(14, 5))
for layers in layer_depths:
    plt.plot(epochs, accuracies_cifar10[layers], marker='o', color=colors[layers], label=f'CIFAR-10 - {layers} layers')
    plt.plot(epochs, accuracies_fashion[layers], marker='x', linestyle='--', color=colors[layers], label=f'Fashion-MNIST - {layers} layers')

plt.title('Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'accuracy_across_epochs.png'), bbox_inches='tight')
plt.show()

# Plot 2: Layer Depth vs. Accuracy
plt.figure(figsize=(7, 5))
for epoch in epochs:
    plt.plot(layer_depths, [accuracies_cifar10[depth][epochs.index(epoch)] for depth in layer_depths],
             marker='o', color=colors[layer_depths[epochs.index(epoch) % len(layer_depths)]], label=f'CIFAR-10 - {epoch} epochs')
    plt.plot(layer_depths, [accuracies_fashion[depth][epochs.index(epoch)] for depth in layer_depths],
             marker='x', linestyle='--', color=colors[layer_depths[epochs.index(epoch) % len(layer_depths)]], label=f'Fashion-MNIST - {epoch} epochs')

plt.title('Layer Depth vs. Accuracy')
plt.xlabel('Layer Depth')
plt.ylabel('Accuracy (%)')
plt.xticks(layer_depths)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'layer_depth_vs_accuracy.png'), bbox_inches='tight')
plt.show()

# Plot 3: Training Time vs. Accuracy
plt.figure(figsize=(7, 5))
for layers in layer_depths:
    plt.plot(training_times_cifar10[layers], accuracies_cifar10[layers], marker='o', color=colors[layers], label=f'CIFAR-10 - {layers} layers')
    plt.plot(training_times_fashion[layers], accuracies_fashion[layers], marker='x', linestyle='--', color=colors[layers], label=f'Fashion-MNIST - {layers} layers')

plt.title('Training Time vs. Accuracy')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'training_time_vs_accuracy.png'), bbox_inches='tight')
plt.show()
