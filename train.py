import numpy as np
import pandas as pd
import torch
import time
import datetime


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.early_stopped = False

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopped = True
                return True
        return False

def evaluate(model, data_loader, device, is_last_epoch=False):
    """
    Calculate classification error (%) for given model
    and data set.

    Parameters:

    - model: A Trained Pytorch Model
    - data_loader: A Pytorch data loader object
    """


    y_true = np.array([], dtype=int)
    y_pred = np.array([], dtype=int)
    conf_matrix = 1
    if is_last_epoch:

        conf_matrix = np.array([[0]*10]*10, dtype=object)
        
        

        

    

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true = np.append(y_true, labels.cpu().numpy())
            y_pred = np.append(y_pred, predicted.cpu().numpy())

            if is_last_epoch:
                for i in range(len(labels)):
                    conf_matrix[labels[i]][predicted[i]] += 1


    error = np.sum(y_pred != y_true) / len(y_true)

    return error, conf_matrix



def train(model, train_loader, test_loader, criterion,
          optimizer, scheduler=None, MODEL_PATHS=None):
    """
    End to end training as described by the original resnet paper:
    https://arxiv.org/abs/1512.03385

    Parameters
    ----------------

    - model: The PyTorch model to be trained
    - n:   Determines depth of the neural network
           as described in paper
    - train_loader:
           PyTorch dataloader object for training set
    - test_loader:
           PyTorch dataloader object for test set
    """
    start_time = time.time()
    # Run on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Training loop
    # -------------------------------
    cols       = ['epoch', 'train_loss', 'train_err', 'test_err']
    results_df = pd.DataFrame(columns=cols).set_index('epoch')

    # classes = ['index','plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print('Epoch \tBatch \tNLLLoss_Train')

    early_stopper = EarlyStopper(patience=3, min_delta=10)
    for epoch in range(100):  # loop over the dataset multiple times
        if epoch < 20:
            MODEL_PATH = MODEL_PATHS[0]
            model.set_folder(MODEL_PATH)
        elif epoch < 50:
            MODEL_PATH = MODEL_PATHS[1]
            model.set_folder(MODEL_PATH)
        else:
            MODEL_PATH = MODEL_PATHS[2]
            model.set_folder(MODEL_PATH)


        model.train()
        running_loss  = 0.0
        best_test_err = 1.0
        for i, data in enumerate(train_loader, 0):   # Do a batch iteration

            # get the inputs
            inputs, labels = data
            
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print average loss for last 50 mini-batches
            running_loss += loss.item()
            if i % 50 == 49:
                print('%d \t%d \t%.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        if scheduler:
            scheduler.step()

        # Record metrics
        model.eval()
        train_loss = loss.item()
        if epoch in [19, 49, 99]:
            train_err, conf_matrix = evaluate(model, train_loader, device, is_last_epoch=True)
            test_err, conf_matrix = evaluate(model, test_loader, device, is_last_epoch=True)
        else:
            train_err, _ = evaluate(model, train_loader, device)
            test_err, conf_matrix = evaluate(model, test_loader, device)
        results_df.loc[epoch] = [train_loss, train_err, test_err]
        
        
        results_df.to_csv(MODEL_PATH + "_results.csv")
        print(f'train_err: {train_err} test_err: {test_err}')
        print()
        if epoch in [19, 49, 99]:
            print("Confusion Matrix: ")
            print(conf_matrix)
            with open(MODEL_PATH + "_conf_matrix.txt", "w") as f:
                f.write(str(conf_matrix))
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")
            with open("pretrained/" + MODEL_PATH + "time.txt", "w") as f:
                f.write(f"Time taken: {datetime.datetime.fromtimestamp(end_time - start_time).strftime('%H:%M:%S')}") 
        
        if early_stopper.early_stop(train_err):
            print(f"Early Stopped at epoch {epoch}")
            break

        # Save best model
        
        if MODEL_PATH and (test_err < best_test_err):
            torch.save(model.state_dict(), MODEL_PATH + ".pt")
            best_test_err = test_err
        
        

    if early_stopper.early_stopped:
        if MODEL_PATH and (test_err < best_test_err):
            torch.save(model.state_dict(), MODEL_PATH + ".pt")


    print('Finished Training')
    model.eval()
    return model
