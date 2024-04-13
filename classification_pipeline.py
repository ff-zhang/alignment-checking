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

    # Check if the last layer is a sigmoid layer
    if model_config.layers[-1]["act"] == "Sigmoid":
        logits_loss = False
    else:
        logits_loss = True

    training_config.set_logits_loss(logits_loss)

    # Create a temporary target vector where everything equal to i is 1 and the rest are 0
    temp_target = t

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
    logits_loss = training_config.get_logits_loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if logits_loss:
        criterion = nn.BCEWithLogitsLoss()
    else:
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

    train_losses.append(criterion(model(X_train), t_train).item())
    val_losses.append(criterion(model(X_val), t_val).item())

    train_accuracies.append(accuracy(model(X_train), t_train, logits_loss))
    val_accuracies.append(accuracy(model(X_val), t_val, logits_loss))

    print("Epoch ", -1, " Train Loss: ", train_losses[-1], " Train Accuracy: ", train_accuracies[-1],
          " Val Loss: ", val_losses[-1], " Val Accuracy: ", val_accuracies[-1])

    for epoch in range(epochs):
        model.train()
        print("Epoch ", epoch, " of ", epochs)
        for i, data in enumerate(train_loader):
            inputs, labels = data
            labels = labels.float()

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            # Log the accuracy
            train_losses.append(loss.item())
            val_losses.append(criterion(model(X_val), t_val).item())

            train_accuracies.append(accuracy(outputs, labels, logits_loss))
            val_accuracies.append(accuracy(model(X_val), t_val, logits_loss))

        full_train_loss = criterion(model(X_train), t_train).item()
        full_val_loss = criterion(model(X_val), t_val).item()
        full_train_accuracy = accuracy(model(X_train), t_train, logits_loss)
        full_val_accuracy = accuracy(model(X_val), t_val, logits_loss)
        print("Epoch ", epoch, " Train Loss: ", full_train_loss, " Train Accuracy: ", full_train_accuracy,
              " Val Loss: ", full_val_loss, " Val Accuracy: ", full_val_accuracy)

    metrics = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }

    return model, metrics


def accuracy(outputs: torch.tensor, labels: torch.tensor, logits_loss: bool) -> float:
    """
    Returns the accruacy of the model given the outputs and labels

    :param logits_loss:
    :param outputs:
    :param labels:
    :return: accuracy
    """

    # If the model uses logits loss, apply a sigmoid to the outputs
    if logits_loss:
        outputs = torch.sigmoid(outputs)

    # Get the predicted class
    predicted = torch.round(outputs)

    # Get the number of correct predictions
    correct = (predicted == labels).sum().item()

    # Return the accuracy
    return correct / labels.size(0)