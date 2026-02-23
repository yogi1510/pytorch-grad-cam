import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

X_train = torch.randn(80, 20)
y_train = torch.randint(0, 2, (80,))
X_test = torch.randn(20, 20)
y_test = torch.randint(0, 2, (20,))

train_X = X_train
train_y = y_train
test_X = X_test
test_y = y_test


class Model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.l1 = nn.Linear(20, 64)
        self.l2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x


net = model()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


def train(data, labels):
    for epoch in range(10):
        for i in range(len(data)):
            optimizer.zero_grad()

            output = net(data[i])
            loss = loss_fn(output, labels[i].float())

            loss.backward()
            optimizer.step()

        print("epoch done")


def evaluate(x, y):
    correct = 0
    for i in range(len(x)):
        out = net(x[i])
        if out > 0.5:
            pred = 1
        else:
            pred = 0
        if pred == y[i]:
            correct += 1
    return correct / len(x)


if __name__ == "__main__":
    train(train_X, train_y)

    acc = evaluate(test_X, test_y)

    print("Accuracy:", acc)

