# Stuðst við https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html fyrir data preprocessing
###
# Þessi kóði sækir gögnin Fashion-MNIST og augmentar þau býr til training og test set
###

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

def repeat_channels(x):
      return x.repeat(3,1,1)


if __name__ == "__main__":
    #Bý til augmentations með transforms.compose
    data_augment = transforms.Compose([
        transforms.RandomRotation(40), #Sný sumum myndum um 40 gráður
        transforms.RandomHorizontalFlip(p=0.5), #Sný sumum myndum um láréttan ás
        transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5), #Breytti litum í myndunum
        transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5), #Breytti fókus 
        transforms.ToTensor(), #breytti myndum i tensor
        transforms.Lambda(repeat_channels),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Norma myndirnar   
    ])

    #Stjórnar batch size
    batch_size = 100

    #Bý til traning set þetta downloadar gögnunum líka
    training_set = torchvision.datasets.FashionMNIST(root='./data', train=True,download=True, transform=data_augment)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)

    #Bý til test set þetta downloadar gögnunum líka
    testing_set = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True, transform=data_augment)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size,shuffle=False, num_workers=2)

    #Bý til classes sem heldur utanum öll classes í gögnum
    classes = training_set.classes

    ### Þessi kóði sýnir random myndir úr training set
    
    # get some random training images
    dataiter = iter(training_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
