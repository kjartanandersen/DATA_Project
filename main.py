from train_test_func import train_net
from DatasetPicker import DatasetPicker
# import time
# import datetime
import os

if __name__ == '__main__':
    # start = time.time()
    ns = [7]
    plain = False
    train_dataset = DatasetPicker.CIFAR10
    test_dataset = DatasetPicker.CIFAR10

    for n in ns:

        model_folder = f'{"plain_" if plain else ""}resnet_{6*n+2}_{train_dataset.name}_{test_dataset.name}'
        if not os.path.exists("pretrained/" + model_folder):
            os.makedirs("pretrained/" + model_folder)

    for n in ns:
        train_net(n=n, train_dataset=train_dataset, test_dataset=test_dataset)


    """ end = time.time()
    print(f"Time taken: {end - start}")
    with open("pretrained/" + model_folder + "/time.txt", "w") as f:
        f.write(f"Time taken: {datetime.datetime.fromtimestamp(end - start).strftime('%H:%M:%S')}") """