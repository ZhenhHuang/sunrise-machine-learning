import matplotlib.pyplot as plt
import numpy as np


def kernel(x1, x2, l=0.5, sigma=0.2):
    m, n = x1.shape[0], x2.shape[0]
    dist = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist[i, j] = np.sum((x1[i] - x2[j]) ** 2)
    return sigma ** 2 * np.exp(-0.5 / l ** 2 * dist)


def predict(trian_x, test_x, train_y, sigma=1e-8):
    K = kernel(trian_x, trian_x)
    cov_prior = K + sigma * np.eye(K.shape[0])
    L = np.linalg.cholesky(cov_prior)
    alpha = np.linalg.solve(L, train_y)
    alpha = np.linalg.solve(L.T, alpha)
    pred_mean = kernel(trian_x, test_x).T @ alpha
    v = np.linalg.solve(L, kernel(trian_x, test_x))
    pred_var = kernel(test_x, test_x) - v.T @ v + sigma * np.eye(v.shape[1])
    return pred_mean, pred_var


train_x = np.array([3, 1, 4, 5, 7, 9])
train_y = np.cos(train_x)*0.3 + np.random.normal(0, 1e-4, size=train_x.shape)
cov = kernel(np.linspace(0, 10, 100), np.linspace(0, 10, 100), l=2.)
sample_size = 3
sample_from_x = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=sample_size)
test_x = np.linspace(0, 10, 100)
pred_mean, pred_var = predict(train_x, test_x, train_y)
uncertainty = 1.96 * np.sqrt(np.diag(cov))

plt.subplot(2, 1, 1)
plt.ylim(-0.5, 0.5)
plt.plot(np.linspace(0, 10, 100), np.zeros(100), label='mean')
plt.legend()
plt.fill_between(np.linspace(0, 10, 100), np.zeros(100) - uncertainty, np.zeros(100) + uncertainty, alpha=0.1)
for i in range(sample_size):
    plt.plot(np.linspace(0, 10, 100), sample_from_x[i], linestyle='--')

plt.subplot(2, 1, 2)
plt.ylim(-0.5, 0.5)
sample_from_x = np.random.multivariate_normal(pred_mean, pred_var, size=sample_size)
for i in range(sample_size):
    plt.plot(test_x, sample_from_x[i], linestyle='--')
plt.scatter(train_x, train_y, c="red", marker="x")
uncertainty = 1.96 * np.sqrt(np.diag(pred_var))
plt.plot(test_x, pred_mean, label='predict', c='blue')
plt.fill_between(test_x, pred_mean - uncertainty, pred_mean + uncertainty, alpha=0.1)
plt.legend()
plt.show()
