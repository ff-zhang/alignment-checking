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

LRP stuff: https://github.com/fhvilshoj/TorchLRP
'''
import torch
import torch.nn as nn
import lrp

class ModelConfig:

    def __init__(self, format: dict):
        assert (format["num_layers"] == len(format["layers"]))

        self.num_layers = format["num_layers"]
        self.layers = []

        for modelLayer in format["layers"]:
            assert (modelLayer["act"] in {"ReLU", "Tanh", "Sigmoid"})

            temp = {
                "layer_type": modelLayer["layer_type"],
                "in_dim": modelLayer["in_dim"],
                "out_dim": modelLayer["out_dim"],
                "dropout": modelLayer["dropout"],
                "act": modelLayer["act"],
                "batch_norm": modelLayer["batch_norm"]
            }

            self.layers.append(temp)

        self.model = None

    def build(self):
        self.model = Model(modelConfig=self, lrp_bool=True)
        return self.model
    
    # def lrp(self):
    #     return LRPModel(self.model)


class Model(nn.Module):
    layer_type_map = {
        "Linear": nn.Linear,
        "Linear_LRP": lrp.Linear
    }

    act_map = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid
    }

    def __init__(self, modelConfig, lrp_bool=False):
        # Ensure that lrp_bool is true iff layer_type is an LRP layer if applicable (applies to Linear rn)
        super(Model, self).__init__()

        self.layers = []
        for i in range(modelConfig.num_layers):
            temp = modelConfig.layers[i]

            is_bn = temp["batch_norm"]

            if lrp_bool:
                if is_bn:
                    layer = lrp.Sequential(
                        Model.layer_type_map[temp["layer_type"]](in_features=temp["in_dim"], out_features=temp["out_dim"]),
                        Model.act_map[temp["act"]](),
                        nn.Dropout(p=temp["dropout"]),
                        nn.BatchNorm1d(temp["in_dim"])
                    )
                else:
                    layer = lrp.Sequential(
                        Model.layer_type_map[temp["layer_type"]](in_features=temp["in_dim"], out_features=temp["out_dim"]),
                        Model.act_map[temp["act"]](),
                        nn.Dropout(p=temp["dropout"])
                    )
            else:
                if is_bn:
                    layer = nn.Sequential(
                        Model.layer_type_map[temp["layer_type"]](in_features=temp["in_dim"], out_features=temp["out_dim"]),
                        Model.act_map[temp["act"]](),
                        nn.Dropout(p=temp["dropout"]),
                        nn.BatchNorm1d(temp["in_dim"])
                    )
                else:
                    layer = nn.Sequential(
                        Model.layer_type_map[temp["layer_type"]](in_features=temp["in_dim"], out_features=temp["out_dim"]),
                        Model.act_map[temp["act"]](),
                        nn.Dropout(p=temp["dropout"])
                    )

            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        # assume X is already in the correct shape and all that
        out = x

        for layer in self.layers:
            out = layer(out)

        return out


if __name__ == "__main__":
    format = {
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

    test_config = ModelConfig(format)
    print(test_config.build())