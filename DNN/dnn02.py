import torch
import pylab as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        
class DNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层
        self.fc1 = nn.Linear(2, 7)
        # 第一层隐藏层
        self.fc2 = nn.Linear(7, 6)
        # 第二层隐藏层
        self.fc3 = nn.Linear(6, 1)

        self.sigmoid = nn.Sigmoid()

    # 正向传播
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层
        y = self.sigmoid(self.fc3(x))
        return y

    # 损失函数
    def loss_func(self, y_pred, y_true):
        return nn.BCELoss()(y_pred, y_true)

    # 优化器
    @property
    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def train_step(model, features, labels):
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions, labels)

    # 反向传播求梯度
    loss.backward()

    # 更新模型参数
    model.optimizer.step()
    model.optimizer.zero_grad()

    return loss.item()


def train_model(model: DNNModel, epochs):
    for epoch in range(1, epochs + 1):
        loss_list = []
        for features, labels in dl:

            # features = features.to(device)
            # labels = labels.to(device)

            lossi = train_step(model, features, labels)
            loss_list.append(lossi)
        loss = np.mean(loss_list)

        if epoch % 10 == 0:
            print("epoch =", epoch, "loss = ", loss)


if __name__ == '__main__':
    # 正负样本数量
    n_positive, n_negative = 2000, 2000

    # 生成正样本, 小圆环分布
    r_p = 5.0 + torch.normal(0.0, 1.0, size=[n_positive, 1])
    theta_p = 2 * np.pi * torch.rand([n_positive, 1])
    Xp = torch.cat([r_p * torch.cos(theta_p), r_p * torch.sin(theta_p)], axis=1)
    Yp = torch.ones_like(r_p)

    # 生成负样本, 大圆环分布
    r_n = 8.0 + torch.normal(0.0, 1.0, size=[n_negative, 1])
    theta_n = 2 * np.pi * torch.rand([n_negative, 1])
    Xn = torch.cat([r_n * torch.cos(theta_n), r_n * torch.sin(theta_n)], axis=1)
    Yn = torch.zeros_like(r_n)

    # 汇总样本
    X = torch.cat([Xp, Xn], axis=0)
    Y = torch.cat([Yp, Yn], axis=0)

    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=10, shuffle=True, num_workers=1)

    model = DNNModel()
    # model = model.to(device)
    train_model(model, 100)

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax1.scatter(Xp[:, 0], Xp[:, 1], c="r")
    ax1.scatter(Xn[:, 0], Xn[:, 1], c="g")
    ax1.legend(["positive", "negative"]);
    ax1.set_title("y_true");

    Xp_pred = X[torch.squeeze(model.forward(X) >= 0.5)]
    Xn_pred = X[torch.squeeze(model.forward(X) < 0.5)]

    ax2.scatter(Xp_pred[:, 0], Xp_pred[:, 1], c="r")
    ax2.scatter(Xn_pred[:, 0], Xn_pred[:, 1], c="g")
    ax2.legend(["positive", "negative"])
    ax2.set_title("y_pred")
    plt.plot()
