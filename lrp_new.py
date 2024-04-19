from innvestigator import InnvestigateModel
from argparse import ArgumentParser
import pickle
import torch
import numpy as np


def load_model_by_index(models_path, index):
    models = pickle.load(open(models_path, "rb"))
    model = models.get(index)
    if model is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    return model


def get_test_dataloader(test_data_path):
    # TODO - create dataloader for the test data
    # idk how the data is stored, so someone has to implement this pls
    pass


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

    args = parser.parse_args()

    # load the model using the provided path and index
    model = load_model_by_index(args.models_path, args.index)
    if model is not None:
        print("Model loaded successfully.")
    else:
        print(f"No model found at index {args.index}")

    dataloader = get_test_dataloader(args.test_data_path)
    if dataloader is not None:
        print("Dataloader loaded successfully.")
    else:
        print(f"Incorrect dataloader filepath.")



