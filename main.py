import pickle
import pandas as pd

from classification_pipeline import *
from model_config import *

if __name__ == "__main__":
    # First thing to do is load the data
    data = pickle.load(open("./glove/kmeans_centers_550.pkl", "rb"))

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

    training_config = TrainingConfig(0.01, 100, 32)

    X = data[0]
    labels = data[1]

    print(type(X))
    print(type(labels))

    # Construct the set of unique labels
    unique_labels = set(labels)

    models = {}

    for k in unique_labels:
        # Create a copy of X
        temp_X = X.clone()

        # Label all the data points with 1 if they are of class k, 0 otherwise
        temp_target = torch.tensor([1 if x == k else 0 for x in labels])

        # Construct the model
        model = construct_classifier(temp_X, temp_target, k, model_config, training_config, plot=True)

        # Add to the model dictionary
        models[k] = model
