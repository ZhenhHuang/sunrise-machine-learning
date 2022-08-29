import os
import numpy as np
import pandas as pd
import pylab as plt
import torch
from torch import nn
import datetime
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# model = nn.Linear(2, 1)


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def loss_func(self, y_pred, y_true):
        return nn.MSELoss()(y_pred, y_true)

    @property
    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

# loss_func = nn.MSELoss()
#
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)/


def train_step(model:LinearRegression, features, labels):
    pred = model(features)
    loss = model.loss_func(pred, labels)
    loss.backward()

    # with torch.no_grad():
    #     model.w -= 0.01 * model.w.grad
    #     model.b -= 0.01 * model.b.grad

    #     model.w.grad.zero_()
    #     model.b.grad = None

    model.optimizer.step()
    model.optimizer.zero_grad()

    return loss.item()


def train_model(model:LinearRegression, epochs):
    for epoch in range(1, epochs + 1):
        loss_all = []
        for features, labels in dl:
            loss = train_step(model, features, labels)
            loss_all.append(loss)
        if epoch % 10 == 0:
            print("epoch = ", epoch, "loss =", np.mean(loss_all))

            # w = model.weight
            # b = model.bias

            # print("model.w", w)
            # print("model.b", b)


if __name__ == '__main__':
    n = 400

    # 生成测试用数据集
    X = 10 * torch.rand([n, 2]) - 5.0  # torch.rand是均匀分布
    w0 = torch.tensor([[2.0], [-3.0]])
    b0 = torch.tensor([[10.0]])
    Y = X @ w0 + b0 + torch.normal(0, 0.01, size=[n, 1])  # @表示矩阵乘法,增加正态扰动

    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=10, shuffle=True, num_workers=1)

    model = LinearRegression()

    train_model(model, epochs=10)

    print(model.parameters())
    # plt.clf()
    # w, b = model.weight, model.bias
    #
    # plt.figure(figsize=(12, 5))
    # ax1 = plt.subplot(121)
    # ax1.scatter(X[:, 0], Y[:, 0], c="b", label="samples")
    # ax1.plot(X[:, 0], w[0, 0] * X[:, 0] + b[0], "-r", linewidth=5.0, label="model")
    # ax1.legend()
    # plt.xlabel("x1")
    # plt.ylabel("y", rotation=0)
    #
    # ax2 = plt.subplot(122)
    # ax2.scatter(X[:, 1], Y[:, 0], c="g", label="samples")
    # ax2.plot(X[:, 1], w[0, 1] * X[:, 1] + b[0], "-r", linewidth=5.0, label="model")
    # ax2.legend()
    # plt.xlabel("x2")
    # plt.ylabel("y", rotation=0)
    #
    # plt.show()
