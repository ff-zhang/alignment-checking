import torch
import pickle

from matplotlib import pyplot as plt


class SeparationLoss(torch.nn.Module):
    def __init__(self):
        super(SeparationLoss, self).__init__()

    def forward(self, x, t, y):
        # Y is a set of vectors that spans a k-subspace
        # y is a tensor of shape d x k where d is the dimension of the vectors and k is the number of vectors
        # Split x into 2 parts

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
        super(Projector, self).__init__()
        self.fc1 = torch.nn.Linear(n * d, n * d * n * d)
        self.fc2 = torch.nn.Linear(n * d * n * d, n * d * k)
        self.fc3 = torch.nn.Linear(n * d * k, d * k)
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

    plot = True

    # Load the data about the clusters
    _, data = pickle.load(open("./glove/kmeans_clusters_500.pkl", "rb"))

    # Check if GPU(s) are available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, labels = None, None
    for k in data.keys():
        if X is None:
            X = torch.Tensor(data[k])
            labels = torch.reshape(torch.Tensor([k] * len(data[k])), (-1, 1))
        else:
            X = torch.cat([X, torch.Tensor(data[k])])
            temp = torch.reshape(torch.Tensor([k] * len(data[k])), (-1, 1))
            labels = torch.cat([labels, temp])

    # Create the set of unique labels
    targets = [x.item() for x in labels]
    unique_labels = list(set(targets))

    # One hot encode the targets for label 0
    target = torch.tensor([1 if x == unique_labels[0] else 0 for x in targets]).to(device) # This is to test with only the first cluster

    # Create the model
    model = Projector(550, 50, 3).to(device)

    # Create the loss function
    criterion = SeparationLoss().to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    # Train the model
    for epoch in range(100):
        optimizer.zero_grad()
        # Stack X into a single tensor of n x d
        X = X.to(device)
        target = target.to(device)

        input = X.view(-1, 550 * 50)

        output = model(input)
        loss = criterion(output, target, X)
        losses.append(loss)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss: {loss.item()}")

        if plot:
            plt.plot(losses)
            plt.show()

    # Save the model
    torch.save(model.state_dict(), "./projector.pth")
