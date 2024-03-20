import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.functional import F

from sklearn.model_selection import train_test_split


# Given the data as a matrix X with

def construct_classifier(X, t, k, model_config, training_config):
    """
    Given the data as a matrix X of shape n x m, an array of cluster/class assignments t of shape n x 1,
    the model configuration, and a training configuration where n is the number of
    data points, m is the shape of each data point, and k is the target class for the classifier,
    returns a binary classifier for this class.


    :input: Matrix X of shape n x m, Array t of shape n x 1

    :return: A k x 1 array of binary classification models
    """

    # Get all the unique labels from the set of data
    labels = set(list(t))

    num_classes = len(labels)

    # Learn a binary classifier of the config architecture for the k'th class
    model = model_config.build()

    # Create a temporary target vector where everything equal to i is 1 and the rest are 0
    temp_target = torch.tensor([1 if x == k else 0 for x in t])

    # Training loop
    train_model(model, X, t, training_config)

    # Return the model
    return model

def train_model(model, X, t, training_config):

    lr = training_config.lr
    epochs = training_config.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.BCELoss()

    # Split the X and t into training and validation sets
    X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2)

    # Create dataloaders
    train_loader = DataLoader(X_train, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(X_val, batch_size=training_config.batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

