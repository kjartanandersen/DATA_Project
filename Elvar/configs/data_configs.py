from torchvision import transforms, datasets

def repeat_channels(x):
      return x.repeat(3,1,1)

DATASETS = {
    'cifar10': {
        'data_dir': '~/data/cifar10',
        'num_classes': 10,
        'image_size': 32,
        'means': [0.4918687901200927, 0.49185976472299225, 0.4918583862227116],
        'stds': [0.24697121702736, 0.24696766978537033, 0.2469719877121087],
        'labels': datasets.CIFAR10('~/.pytorch/CIFAR_data/', download=True, train=True).classes,
        'dataset_class': datasets.CIFAR10,
        'transform_train': lambda: transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5),
            transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        'transform_test': lambda: transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5),
            transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ])
    },
    'fashionMNIST': {
        'data_dir': 'data/FashionMNIST',
        'num_classes': 10,
        'image_size': 28,
        'means': [-0.42791882, -0.42791882, -0.42791882],
        'stds': [0.25211963, 0.25211963, 0.25211963],
        'labels': datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/',download=True,train=True),
        'dataset_class': datasets.FashionMNIST,
        'transform_train': lambda: transforms.Compose([
            transforms.Lambda(repeat_channels),
            transforms.Pad(4),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5),
            transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
        ]), 
        'transform_test': lambda: transforms.Compose([
            transforms.Lambda(repeat_channels),
            transforms.Pad(4),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness = 0.2,contrast = 0.1,saturation = 0.5),
            transforms.RandomAdjustSharpness(sharpness_factor = 1,p = 0.5), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
        ])
    }
}