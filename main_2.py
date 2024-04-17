import torch
import pickle

from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import pairwise_distances
import numpy as np

import os

class SeparationLoss(torch.nn.Module):
    def __init__(self, k):
        super(SeparationLoss, self).__init__()
        self.k = k

    def forward(self, x, t, y):
        # Y is a set of vectors that spans a k-subspace
        # y is a tensor of shape d x k where d is the dimension of the vectors and k is the number of vectors
        # Split x into 2 parts

        k = self.k

        #split y into d x k
        y = y.view(d, k)

        vecs = x
        labels = t

        # Project the vectors onto the k subspace
        # y is a matrix of shape d x k
        # vecs is a matrix of shape n x d
        # n is the number of vectors
        # The projection is vecs @ y
        proj_matrix = y @ (y.T @ y).inverse() @ y.T

        # The projection of the vectors
        proj_vecs = proj_matrix @ vecs.T  # Transpose to get the d x n matrix
        # This product produces a matrix of shape d x n of the projected vectors

        # Transpose the projected vectors to get a matrix of shape n x d
        proj_vecs = proj_vecs.T

        # Measure how distinct the projected vectors are based on their label
        # Want to minimize distance between projected vectors of the same label and maximize distance between projected vectors of different labels

        loss = 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                dist = torch.norm(proj_vecs[i] - proj_vecs[j])
                if labels[i] == labels[j]:
                    loss += dist
                else:
                    loss -= dist

        return loss

class Projector(torch.nn.Module):
    """
    This class takes in a vector of dimension n x d where n is the number of vectors
    and outputs a vector of size d x k where k is the number of vectors
    that span a <=k-subspace
    """
    def __init__(self, n, d, k):
        self.n = n
        self.d = d
        self.k = k
        super(Projector, self).__init__()
        self.fc1 = torch.nn.Linear(n * d, n * d)
        self.fc2 = torch.nn.Linear(n * d, n * k)
        self.fc3 = torch.nn.Linear(n * k, d * k)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(2)
    batch_size = 128
    n = batch_size
    d = 50
    k = 3
    lr = 0.0001
    epochs = 5

    plot = False

    # Load the data about the clusters
    _, data = pickle.load(open("./glove/kmeans_clusters_500.pkl", "rb"))

    # Check if GPU(s) are available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, labels = None, None
    for key in data.keys():
        if X is None:
            X = torch.Tensor(data[key])
            labels = torch.reshape(torch.Tensor([key] * len(data[key])), (-1, 1))
        else:
            X = torch.cat([X, torch.Tensor(data[key])])
            temp = torch.reshape(torch.Tensor([key] * len(data[key])), (-1, 1))
            labels = torch.cat([labels, temp])

    # Create the set of unique labels
    targets = [x.item() for x in labels]
    unique_labels = list(set(targets))

    for j in range(len(unique_labels)):
        print("Training for label", j)
        # One hot encode the targets for label 0
        # This is to test with only the first cluster
        target = torch.tensor([1 if x == unique_labels[j] else 0 for x in targets])

        # Create the model
        model = Projector(n, d, k).to(device)

        # Create the loss function
        criterion = SeparationLoss(k).to(device)

        # Create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = []

        # Collect the entries of X with label j
        X_pad = torch.tensor([])
        for i in range(len(X)):
            if target[i] == 1:
                X_pad = torch.cat([X_pad, X[i].reshape(1, -1)])

        batch_remainder = len(X) % batch_size
        padding = torch.tensor([])
        for i in range(batch_size - batch_remainder):
            padding = torch.cat([padding, (X_pad[i]).reshape(1, -1)])

        train_data = TensorDataset(X, target)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

        if os.path.exists(f"./projector-{j}.pth"):
            model.load_state_dict(torch.load(f"./projector-{j}.pth"))
        else:
            # Train the model
            for epoch in range(epochs):
                optimizer.zero_grad()
                for X, t in train_loader:
                    if X.shape[0] != batch_size:
                        # Pad x with the padding
                        X = torch.cat([X, padding])
                    X = X.to(device)
                    t = t.to(device)
                    _X = X.view(-1, batch_size * d)

                    output = model(_X)
                    loss = criterion(X, t, output)
                    print(f"Epoch [{epoch}]: Loss: {loss.item()}")
                    losses.append(loss)
                    loss.backward()
                    optimizer.step()

                print(f"\n Epoch Complete {epoch}: Loss: {loss.item()} \n")

            if plot:
                for i in range(len(losses)):
                    losses[i] = losses[i].detach().numpy()
                plt.plot(losses)
                plt.show()

            # Save the model
            torch.save(model.state_dict(), f"./projector-{j}.pth")

        # Test

        # Run through the entire loader and collect the outputs
        outputs = []
        for X, t in train_loader:
            if X.shape[0] != batch_size:
                # Pad x with the first entry
                X = torch.cat([X, padding])
            X = X.to(device)
            t = t.to(device)
            _X = X.view(-1, batch_size * d)
            output = model(_X)
            # Reshape the output to be of shape k x d
            output = output.view(k, d)
            outputs.append(output)

        # Average the outputs
        avg_output = torch.sum(torch.stack(outputs), dim=0)

        # load 50d glove embeddings
        with open('glove/glove_50d.pkl', 'rb') as f:
            glove_50d = pickle.load(f)

        print(len(glove_50d))

        # reverse the mapping
        glove_50d_inv = {tuple(v): k for k, v in glove_50d.items()}

        glove_50d_data = np.array(list(glove_50d.values()))
        glove_50d_labels = np.array(list(glove_50d.keys()))

        print("Closest words for class ", j, ":")
        dict = {}
        for i in range(0, k):
            distances = pairwise_distances([avg_output[i].detach().numpy()], glove_50d_data, metric='cosine')
            closest_words = np.argsort(distances)[0][:1]
            print("Closest words to dimension", i, "is:")
            for word in glove_50d_labels[closest_words]:
                print(word)
            print("at distance", distances[0][closest_words])
            dict[i] = glove_50d_labels[closest_words]

        # Pickle the dictionary
        with open(f"./closest_words-{j}.pkl", "wb") as f:
            pickle.dump(dict, f)