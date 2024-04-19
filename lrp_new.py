import os

from innvestigator import InnvestigateModel
from argparse import ArgumentParser
import pickle
import torch
import numpy as np
import torch.utils.data

import pickle
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_model_by_index(models_path, index):
    models = pickle.load(open(models_path, "rb"))
    model = models.get(index)
    if model is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    return model


def get_test_dataloader(test_data_path, target_class):
    # TODO - create dataloader for the test data
    # idk how the data is stored, so someone has to implement this pls
    words, vectors = pickle.load(open(test_data_path, "rb"))

    X = torch.tensor([])
    targets = torch.tensor([])

    for cls in vectors.keys():
        X = torch.cat((X, vectors[cls]))
        if cls == target_class:
            targets = torch.cat((targets, torch.tensor([1] * len(vectors[cls]))))
        else:
            targets = torch.cat((targets, torch.tensor([0] * len(vectors[cls]))))

    # create a dataloader from the data

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, targets), batch_size=64, shuffle=True)

    return loader


# reference: https://github.com/moboehle/Pytorch-LRP/blob/master/MNIST%20example.ipynb
def do_lrp(model, dataloader, device):
    inn_model = InnvestigateModel(model, lrp_exponent=2,
                                  method="e-rule",
                                  beta=.5)

    # assume all predictions happen for one class / from one binary model
    evidence_for_class = []

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        batch_size = int(data.size()[0])

        model_prediction, true_relevance = inn_model.innvestigate(in_tensor=data)

        # Below is code from the reference above:
        # for i in range(10):
        #     # Unfortunately, we had some issue with freeing pytorch memory, therefore
        #     # we need to reevaluate the model separately for every class.
        #     model_prediction, input_relevance_values = inn_model.innvestigate(in_tensor=data, rel_for_class=i)
        #     evidence_for_class.append(input_relevance_values)

        # since ours deals with individual models for each class, need to figure out how to determine which class it is
        # for now, rel_for_class is None => the 'winning' class is used for indexing
        model_pred, input_rel_values = inn_model.innvestigate(in_tensor=data)
        evidence_for_class.append(input_rel_values)

    evidence_for_class = np.array([elt.numpy() for elt in evidence_for_class])

    # the example contained plots of relevance visualization on MNIST dataset
    # idk how to visualize it for the data we have

    return evidence_for_class


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--models_path", type=str, required=True, help="Path to the serialized file containing models.")
    parser.add_argument("--index", type=int, required=True, help="Index of the model to load.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Filepath for the test dataset.")
    parser.add_argument("--target_class", type=int, required=True, help="Class to generate LRP for.")

    args = parser.parse_args()

    if not os.path.exists(args.models_path):
        with open('sharding.pkl', 'rb') as f:
            interval_dict = pickle.load(f)
        models = {}
        interval = 25

        for k in interval_dict.keys():
            start = k
            end = start + interval

            if not torch.cuda.is_available():
                shard = CPU_Unpickler(open(f'models_{start}_{end}.pkl', 'rb')).load()
            else:
                shard = pickle.load(open(f'models_{start}_{end}.pkl', 'rb'))

            for key in shard.keys():
                if shard[key] is not None:
                    models[key] = shard[key].to('cpu')

        # Save the models dictionary to a file
        with open(args.models_path, 'wb') as f:
            pickle.dump(models, f)

    k = args.target_class

    # load the model using the provided path and index
    model = load_model_by_index(args.models_path, args.index)
    if model is not None:
        print("Model loaded successfully.")
    else:
        print(f"No model found at index {args.index}")

    dataloader = get_test_dataloader(args.test_data_path, k)
    if dataloader is not None:
        print("Dataloader loaded successfully.")
    else:
        print(f"Incorrect dataloader filepath.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evidence = do_lrp(model, dataloader, device) # what to do with the output from this, probably need to figure out how to visualize
    print("Evidence = \n{}".format(evidence))
