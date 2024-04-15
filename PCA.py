import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from model_config import ModelConfig


def extract_pca_features(representation: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
    """
    Extract PCA features from the model given LRP representation of shape N x 1 x H x W

    :param threshold:
    :param representation:
    :return: principal components of the representation
    """

    # Reshape the representation to N x (H * W)
    representation = representation.reshape(representation.shape[0], -1)

    # Scale the representation
    representation = (representation - representation.mean()) / representation.std()

    # Apply PCA to the representation
    pca = PCA(n_components=threshold, svd_solver='full')

    # Fit the PCA model
    pca.fit(representation)

    # Return the principal components
    return pca.components_
