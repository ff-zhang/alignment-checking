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
        assert(format.num_layers == len(format.layers))

        self.num_layers = format.num_layers
        self.layers = []

        for modelLayer in format.layers:
            assert(modelLayer.act in {"ReLU", "Tanh", "Sigmoid"})

            temp = {
                "layer_type": modelLayer.layer_type,
                "in_dim": modelLayer.in_dim,
                "out_dim": modelLayer.out_dim,
                "dropout": modelLayer.dropout,
                "act": modelLayer.act,
                "batch_norm": modelLayer.batch_norm 
            }

            self.layers.append(temp)


    def build(self):
        return Model(self)


class Model(nn.Module):

    layer_type_map = {
        "Linear": nn.Linear
    }
    
    act_map = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid
    }

    def __init__(self, ModelConfig):
        super().__init__()

