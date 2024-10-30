from train_test_func import train_net
from DatasetPicker import DatasetPicker

if __name__ == '__main__':
    train_net(n=3, epochs=20, train_dataset=DatasetPicker.CIFAR10, test_dataset=DatasetPicker.CIFAR10)