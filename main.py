import pickle
import pandas as pd
from copy import deepcopy
import numpy as np
import torch

from classification_pipeline import *
from model_config import *
import os
import lrp

if __name__ == "__main__":
    # First thing to do is load the data
    data = pickle.load(open("./glove/clusters_550.pkl", "rb"))

    batch_size = 32

    # X = data[0]
    data = data[1]

    X = None
    labels = None

    for k in data.keys():
        if X is None:
            X = torch.Tensor(data[k])
            labels = torch.reshape(torch.Tensor([k] * len(data[k])), (-1, 1))
        else:
            X = torch.cat([X, torch.Tensor(data[k])])
            temp = torch.reshape(torch.Tensor([k] * len(data[k])), (-1, 1))
            labels = torch.cat([labels, temp])

    # Construct the set of unique labels
    targets = [x.item() for x in labels]
    unique_labels = list(set(targets))

    model_format = {
        "num_layers": 4,
        "layers": [
            {
                "layer_type": "Linear_LRP",
                "in_dim": 50,
                "out_dim": 64,
                "act": "ReLU",
                "dropout": 0.1,
                "batch_norm": False
            },
            {
                "layer_type": "Linear_LRP",
                "in_dim": 64,
                "out_dim": 128,
                "act": "ReLU",
                "dropout": 0.0,
                "batch_norm": False
            },
            {
                "layer_type": "Linear_LRP",
                "in_dim": 128,
                "out_dim": 32,
                "act": "ReLU",
                "dropout": 0.0,
                "batch_norm": False
            },
            {
                "layer_type": "Linear_LRP",
                "in_dim": 32,
                "out_dim": 1,
                "act": "Sigmoid",
                "dropout": 0.0,
                "batch_norm": False
            },
        ]
    }

    model_config = ModelConfig(model_format)

    # Check if the models have been saved
    if os.path.exists("models.pkl"):
        models = pickle.load(open("models.pkl", "rb"))
        print("Models loaded")
    else:
        print("Models not found")

        training_config = TrainingConfig(0.01, 1, batch_size)

        models = {}

        # k = unique_labels[0]
        for k in unique_labels:
            print("Training model for class", k)
            # Create a copy of X
            temp_X = deepcopy(X)

            # Label all the data points with 1 if they are of class k, 0 otherwise
            temp_target = torch.tensor([1 if x == k else 0 for x in labels])

            # Construct the model
            model = construct_classifier(temp_X, temp_target, k, model_config, training_config, plot=False)

            # Add to the model dictionary
            models[k] = model

            # Save the models
            pickle.dump(models, open("models.pkl", "wb"))

    # Now that we have the models

    if os.path.exists("explanations.pkl"):
        explanations = pickle.load(open("explanations.pkl", "rb"))
        print("Explanations loaded")
    else:
        X.requires_grad = True

        explanations = {}

        if model_config.layers[-1]["act"] == "Sigmoid":
            criterion = nn.BCELoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        # k = unique_labels[0]
        for k in unique_labels:
            print("Explaining model for class", k)
            model = models[k]

            predictions = model.forward(X, explain=True, rule="alpha2beta1")

            pred = predictions.sum()

            loss = criterion(predictions, torch.reshape(torch.tensor([1 if x == k else 0 for x in labels]), (-1, 1)).float())

            pred.backward()

            explanations[k] = X.grad

            # Save the explanations
            pickle.dump(explanations, open("explanations.pkl", "wb"))
