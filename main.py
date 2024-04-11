import pickle
import pandas as pd
from copy import deepcopy
import numpy as np

from classification_pipeline import *
from model_config import *
import os

if __name__ == "__main__":
    # First thing to do is load the data
    data = pickle.load(open("./glove/clusters_550.pkl", "rb"))

    batch_size = 32

    # X = data[0]
    data = data[1]

    X = np.array([])
    labels = np.array([])

    for k in data.keys():
        if X.shape[0] == 0:
            X = np.array(data[k])
        else:
            X = np.concatenate([X, np.array(data[k])])
        labels = np.concatenate([labels, np.array([k] * len(data[k]))])

    # Shuffle the data
    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    X = X[indices]
    labels = labels[indices]

    # Construct the set of unique labels
    unique_labels = set(labels)

    # Check if the models have been saved
    if os.path.exists("models.pkl"):
        models = pickle.load(open("models.pkl", "rb"))
        print("Models loaded")
    else:
        print("Models not found")
        model_format = {
            "num_layers": 2,
            "layers": [
                {
                    "layer_type": "Linear",
                    "in_dim": 3,
                    "out_dim": 64,
                    "act": "ReLU",
                    "dropout": 0.1,
                    "batch_norm": True
                },
                {
                    "layer_type": "Linear",
                    "in_dim": 64,
                    "out_dim": 3,
                    "act": "ReLU",
                    "dropout": 0.0,
                    "batch_norm": False
                },
            ]
        }

        model_config = ModelConfig(model_format)

        training_config = TrainingConfig(0.01, 100, batch_size)

        models = {}

        for k in unique_labels:
            # Create a copy of X
            temp_X = deepcopy(X)

            # Label all the data points with 1 if they are of class k, 0 otherwise
            temp_target = torch.tensor([1 if x == k else 0 for x in labels])

            # Construct the model
            model = construct_classifier(temp_X, temp_target, k, model_config, training_config, plot=True)

            # Add to the model dictionary
            models[k] = model

        # Save the models
        pickle.dump(models, open("models.pkl", "wb"))

    # Now that we have the models

    explanations = {}

    for k in unique_labels:
        model = models[k]

        predictions = model.forward(X, explain=True, rule="alpha2beta1")

        predictions = predictions[torch.arange(batch_size), predictions.max(1)[1]]  # Choose maximizing output neuron

        predictions = predictions.sum()

        predictions.backward()

        explanation = X.grad

        explanations[k] = explanation

    # Save the explanations
    pickle.dump(explanations, open("explanations.pkl", "wb"))