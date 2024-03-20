import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from model_config import ModelConfig


def extract_pca_features(representation: torch.Tensor, model: ModelConfig):
    '''
    Extract PCA features from the model given LRP representation of shape N x 1 x H x W

    :param representation:
    :param model:
    :return: principal components of the representation
    '''

    # Reshape the representation to N x (H * W)
    representation = representation.reshape(representation.shape[0], -1)

    # Scale the representation
    representation = (representation - representation.mean(dim=0)) / representation.std(dim=0)

    # Apply PCA to the representation
    pca = PCA(n_components=0.95, svd_solver='full')

    # Fit the PCA model
    pca.fit(representation)

    # Return the principal components
    return pca.components_
