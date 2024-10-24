import numpy as np
from torchvision import transforms, datasets
from PIL import Image

def repeat_channels(x):
    if isinstance(x, Image.Image):  # If input is a PIL Image
        x = transforms.ToTensor()(x)  # Convert to tensor
    return x.repeat(3, 1, 1)  # Repeat channels

# Load FashionMNIST dataset (train or test, it shouldn't matter much)
fashion_mnist = datasets.FashionMNIST('data/FashionMNIST', train=True, download=True, transform=None)

# Apply transformations to calculate means and stds
transformed_fashion_mnist = datasets.FashionMNIST('data/FashionMNIST', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor first
    transforms.Lambda(repeat_channels),  # Then repeat channels
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize (temporary, will be updated)
]))

# Calculate means and stds
means = []
stds = []
for channel in range(3):
    channel_values = [img[channel, :, :].mean() for img, _ in transformed_fashion_mnist]
    means.append(np.mean(channel_values))
    stds.append(np.std(channel_values))

print("FashionMNIST (RGB) means:", tuple(means))
print("FashionMNIST (RGB) stds:", tuple(stds))

# Now, apply the same transformations with the calculated means and stds for normalization
fashion_mnist = datasets.FashionMNIST('data/FashionMNIST', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(repeat_channels),
    transforms.Pad(4),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5),
    transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5), 
    transforms.Normalize(tuple(means), tuple(stds)),  # Use calculated means and stds
]))