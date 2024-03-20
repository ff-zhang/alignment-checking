import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.functional import F

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# Given the data as a matrix X with

def construct_classifier(X, t, k, model_config, training_config, plot=False):
    """
    Given the data as a matrix X of shape n x m, an array of cluster/class assignments t of shape n x 1,
    the model configuration, and a training configuration where n is the number of
    data points, m is the shape of each data point, and k is the target class for the classifier,
    returns a binary classifier for this class.


    :input: Matrix X of shape n x m, Array t of shape n x 1

    :return: A k x 1 array of binary classification models
    """

    # Learn a binary classifier of the config architecture for the k'th class
    model = model_config.build()

    # Create a temporary target vector where everything equal to i is 1 and the rest are 0
    temp_target = torch.tensor([1 if x == k else 0 for x in t])

    # Training loop
    model, metrics = train_model(model, X, temp_target, training_config)

    train_losses = metrics["train_losses"]
    train_accuracies = metrics["train_accuracies"]
    val_losses = metrics["val_losses"]
    val_accuracies = metrics["val_accuracies"]

    if plot:
        plt.figure()
        plt.title("Loss for Class " + str(k))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.show()

        plt.figure()
        plt.title("Accuracy for Class " + str(k))
        plt.plot(train_accuracies, label="Train")
        plt.plot(val_accuracies, label="Val")
        plt.show()

    # Return the model
    return model


def train_model(model, X, t, training_config, plot=False):
    lr = training_config.lr
    epochs = training_config.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.BCELoss()

    # Split the X and t into training and validation sets
    X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2)

    # Create dataloaders
    train_loader = DataLoader(X_train, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=training_config.batch_size, shuffle=True)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            # Log the accuracy
            train_losses.append(loss)
            val_losses.append(criterion(model(X_val), t_val))

            train_accuracies.append(accuracy(outputs, labels))
            val_accuracies.append(accuracy(model(X_val), t_val))

    metrics = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }

    return model, metrics


def accuracy(outputs, labels):
    return (outputs == labels).sum() / len(labels)
