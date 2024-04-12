import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from model_config import ModelConfig
from training_config import TrainingConfig


def construct_classifier(X: torch.tensor, t: torch.tensor, k: int, model_config: ModelConfig,
                         training_config: TrainingConfig, plot: bool = False) -> nn.Module:
    """
    Given the data as a matrix X of shape n x m, an array of cluster/class assignments t of shape n x 1,
    the model configuration, and a training configuration where n is the number of
    data points, m is the shape of each data point, and k is the target class for the classifier,
    returns a binary classifier for this class.


    :param X: An n x m matrix of data points
    :param t: An n x 1 array of class assignments
    :param k: The target class
    :param model_config: The model configuration
    :param training_config: The training configuration
    :param plot: Whether to plot the training and validation loss and accuracy

    :return: A binary classification model for the k'th class
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


def train_model(model: nn.Module, X: torch.tensor, t: torch.tensor, training_config: TrainingConfig) -> (
nn.Module, dict):
    lr = training_config.get_lr()
    epochs = training_config.get_epochs()
    batch_size = training_config.get_batch_size()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.BCELoss()

    # Split the X and t into training and validation sets
    X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2)
    t_train = torch.reshape(t_train, (-1, 1))
    t_val = torch.reshape(t_val, (-1, 1))

    # Cast to float
    t_train = t_train.float()
    t_val = t_val.float()

    train = []
    for i in range(len(X_train)):
        train.append((X_train[i], t_train[i]))
    val = []
    for i in range(len(X_val)):
        val.append((X_val[i], t_val[i]))

    # Create dataloaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            labels = labels.float()

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


def accuracy(outputs: torch.tensor, labels: torch.tensor) -> float:
    """
    Returns the accruacy of the model given the outputs and labels

    :param outputs:
    :param labels:
    :return: accuracy
    """

    # Get the predicted class
    _, predicted = torch.max(outputs, 1)

    # Get the number of correct predictions
    correct = (predicted == labels).sum().item()

    # Return the accuracy
    return correct / labels.size(0)