"""Trains and evaluates a toy binary classifier on random data for ai_ml_review tests."""
import torch
import torch.nn as nn
import torch.optim as optim

train_X = torch.randn(80, 20)
train_y = torch.randint(0, 2, (80,))
test_X = torch.randn(20, 20)
test_y = torch.randint(0, 2, (20,))


class Model(nn.Module):
    """Two-layer MLP for binary classification."""

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
    with torch.no_grad():
        logits = net(x).squeeze(1)
        preds = (logits > 0.0).long()
        correct = (preds == y).sum().item()
    return correct / len(x)


if __name__ == "__main__":
    train(train_X, train_y)

    acc = evaluate(test_X, test_y)

    print("Accuracy:", acc)

