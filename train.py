import numpy as np
import pandas as pd
import torch

def evaluate(model, data_loader, device):
    """
    Calculate classification error (%) for given model
    and data set.

    Parameters:

    - model: A Trained Pytorch Model
    - data_loader: A Pytorch data loader object
    """

    conf_matrix = np.zeros((10, 10), dtype=int)

    y_true = np.array([], dtype=int)
    y_pred = np.array([], dtype=int)

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true = np.concatenate((y_true, labels.cpu()))
            y_pred = np.concatenate((y_pred, predicted.cpu()))

            conf_matrix += np.bincount(10 * labels.cpu().numpy() + predicted.cpu().numpy(),
                                       minlength=100).reshape(10, 10)

    error = np.sum(y_pred != y_true) / len(y_true)

    return error, conf_matrix



def train(model, epochs, train_loader, test_loader, criterion,
          optimizer, scheduler=None, MODEL_PATH=None):
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

    # Run on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Training loop
    # -------------------------------
    cols       = ['epoch', 'train_loss', 'train_err', 'test_err']
    results_df = pd.DataFrame(columns=cols).set_index('epoch')
    print('Epoch \tBatch \tNLLLoss_Train')

    for epoch in range(epochs):  # loop over the dataset multiple times

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
        train_err, _ = evaluate(model, train_loader, device)
        test_err, conf_matrix = evaluate(model, test_loader, device)
        results_df.loc[epoch] = [train_loss, train_err, test_err]
        results_df.to_csv(MODEL_PATH + "_results.csv")
        print(f'train_err: {train_err} test_err: {test_err}')
        print()
        print("Confusion Matrix:")
        print(conf_matrix)

        # Save best model
        if MODEL_PATH and (test_err < best_test_err):
            torch.save(model.state_dict(), MODEL_PATH + ".pt")
            best_test_err = test_err
        
        if conf_matrix is not None:
            with open(MODEL_PATH + "_conf_matrix.txt", "w") as f:
                f.write(str(conf_matrix))



    print('Finished Training')
    model.eval()
    return model
