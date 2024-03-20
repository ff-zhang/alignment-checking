'''
Model Config: contains a representation of models that we can use to build models

Input: Dictionary
{

    num_layers: int    // number of layers in the model
    layers: [ ModelLayer ]     // list of ModelLayer, list length = num_layers

}

ModelLayer format:
{

    layer_type: one of [Linear, ... (TBD)]
    in_dim: int
    out_dim: int
    dropout: float between 0-1
    act: one of [ReLU, Tanh, Sigmoid]
    batch_norm: True/False

}

Don't be stupid with it, make sure the dimensions match please.
'''
import torch
import torch.nn as nn


# to import, classifcation_pipeline.py

class ModelConfig:

    def __init__(self, format: dict):
        assert (format.num_layers == len(format.layers))

        self.num_layers = format.num_layers
        self.layers = []

        for modelLayer in format.layers:
            assert (modelLayer.act in {"ReLU", "Tanh", "Sigmoid"})

            temp = {
                "layer_type": modelLayer.layer_type,
                "in_dim": modelLayer.in_dim,
                "out_dim": modelLayer.out_dim,
                "dropout": modelLayer.dropout,
                "act": modelLayer.act,
                "batch_norm": modelLayer.batch_norm
            }

            self.layers.append(temp)

        self.model = None

    def build(self):
        self.model = Model(self)
        return self.model


class Model(nn.Module):
    layer_type_map = {
        "Linear": nn.Linear
    }

    act_map = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid
    }

    def __init__(self, modelConfig):
        super().__init__()

        layers = []
        for i in range(modelConfig.num_layers):
            temp = modelConfig.num_layers[i]
            layer = nn.Sequential(
                Model.layer_type_map[temp["layer_type"]](in_features=temp["in_dim"], out_features=temp["out_dim"]),
                Model.act_map[temp["act"]](),
                nn.Dropout(p=temp["dropout"])
            )

            is_bn = temp.batch_norm
            num_feat = temp["in_dim"]

            layers.append((layer, is_bn, num_feat))

    def forward(self, x):
        # assume X is already in the correct shape and all that
        out = x

        for layer in self.layers:
            if layer[1]:
                out = nn.BatchNorm1d(layer[2])
            out = layer[0](out)

        return x
