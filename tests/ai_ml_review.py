"""Trains and evaluates a toy binary classifier on random data for ai_ml_review tests."""
import torch
import torch.nn as nn
import torch.optim as optim

X_train = torch.randn(80, 20)
y_train = torch.randint(0, 2, (80,))
X_test = torch.randn(20, 20)
y_test = torch.randint(0, 2, (20,))

train_X = X_train
train_y = y_train
test_X = X_test
test_y = y_test


class Model(nn.Module):
    """Two-layer MLP for binary classification."""


def train(data, labels, batch_size=32):
    """Train the model for a fixed number of epochs."""
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(20, 64)
        self.l2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x


net = Model()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


# Recommended approach with batching:
def train(data, labels, batch_size=32):
    """Trains the global `net` on the provided dataset.

    Args:
        data (Tensor): Input features shaped (N, 20).
        labels (Tensor): Binary targets shaped (N,).
        batch_size (int, optional): Mini-batch size for the DataLoader.

    Returns:
        None
    """
    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(10):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = net(batch_x).squeeze(1)
            loss = loss_fn(output, batch_y.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done")


def evaluate(x, y):
    """Computes classification accuracy of `net` on the provided tensors.

    Args:
        x (Tensor): Feature matrix shaped (N, 20).
        y (Tensor): Binary labels shaped (N,).

    Returns:
        float: Accuracy in the range [0, 1].
    """
    net.eval()  # Set model to evaluation mode
    correct = 0
    with torch.no_grad():  # Disable gradient computation
        for i in range(len(x)):
            out = net(x[i])
            pred = 1 if out.item() > 0.0 else 0
            if pred == y[i].item():
                correct += 1
    return correct / len(x)


if __name__ == "__main__":
    train(train_X, train_y)

    acc = evaluate(test_X, test_y)

    print("Accuracy:", acc)

