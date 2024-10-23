from torchvision import transforms, datasets

def repeat_channels(x):
      return x.repeat(3,1,1)

DATASETS = {
    'cifar10': {
        'data_dir': 'data/cifar10',
        'num_classes': 10,
        'image_size': 32,
        'means': [0.4918687901200927, 0.49185976472299225, 0.4918583862227116],
        'stds': [0.24697121702736, 0.24696766978537033, 0.2469719877121087],
        'transform_train': lambda: transforms.Compose([  # define train transforms
            transforms.RandomRotation(40), #Sný sumum myndum um 40 gráður
            transforms.RandomHorizontalFlip(p=0.5), #Sný sumum myndum um láréttan ás
            transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5), #Breytti litum í myndunum
            transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5), #Breytti fókus 
            transforms.ToTensor(), #breytti myndum i tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Norma myndirnar   
        ]),
        'transform_test': lambda: transforms.Compose([  # define test transforms
            transforms.RandomRotation(40), #Sný sumum myndum um 40 gráður
            transforms.RandomHorizontalFlip(p=0.5), #Sný sumum myndum um láréttan ás
            transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5), #Breytti litum í myndunum
            transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5), #Breytti fókus 
            transforms.ToTensor(), #breytti myndum i tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Norma myndirnar   
        ]),
        'dataset_class': datasets.CIFAR10
    },
    'fashionMNIST': {
        'data_dir': 'data/FashionMNIST',
        'num_classes': 10,
        'image_size': 28,
        'means': [0.5],  # assuming single-channel (grayscale)
        'stds': [0.5],
        'transform_train': lambda: transforms.Compose([  # define train transforms
            transforms.Lambda(repeat_channels), # Bæti við channelum til að matcha við cifar10
            transforms.Pad(4), # Padda stærðina um 4 pixla til að komast upp í 32
            transforms.RandomRotation(40), #Sný sumum myndum um 40 gráður
            transforms.RandomHorizontalFlip(p=0.5), #Sný sumum myndum um láréttan ás
            transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5), #Breytti litum í myndunum
            transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5), #Breytti fókus 
            transforms.ToTensor(), #breytti myndum i tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Norma myndirnar   
        ]), 
        'transform_test': lambda: transforms.Compose([  # define test transforms
            transforms.Lambda(repeat_channels), # Bæti við channelum til að matcha við cifar10
            transforms.Pad(4), # Padda stærðina um 4 pixla til að komast upp í 32
            transforms.RandomRotation(40), #Sný sumum myndum um 40 gráður
            transforms.RandomHorizontalFlip(p=0.5), #Sný sumum myndum um láréttan ás
            transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5), #Breytti litum í myndunum
            transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5), #Breytti fókus 
            transforms.ToTensor(), #breytti myndum i tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Norma myndirnar   
        ]),
        'dataset_class': datasets.FashionMNIST
    }
}