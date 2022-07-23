import numpy as np
from linear_model_for_regression.basis_function import RBF
from linear_model_for_regression.regression import Regression
import matplotlib.pyplot as plt


def load_data(path):
    f = open(path)
    dataset = []
    for line in f.readlines():
        line = line.strip().split()
        line = list(map(float, line))
        dataset.append(line)
    dataset = np.array(dataset)
    train = dataset[: 150]
    np.random.shuffle(train)
    test = dataset[150:]
    return train[:, 0], train[:, 1], test[:, 0], test[:, 1]


# class LinearRegression(Regression):
#     def __init__(self, w=None):
#         if w is not None:
#             assert isinstance(w, np.ndarray)
#             assert w.ndim == 2
#         self.w = w
#         self.var = None

#     def fit(self, x_train: np.ndarray, y_train: np.ndarray):
#         if x_train.ndim == 1:
#             x_train = x_train[:, None]
#         if y_train.ndim == 1:
#             y_train = y_train[:, None]
#         self.w = np.linalg.pinv(x_train) @ y_train
#         self.var = np.mean(np.square(y_train - x_train @ self.w))

#     def predict(self, x_test: np.ndarray, return_std=False):
#         if x_test.ndim == 1:
#             x_test = x_test[:, None]
#         y = x_test @ self.w
#         if return_std:
#             std = np.sqrt(self.var) + np.zeros_like(y)
#             return y, std
#         return y


class RidgeRegression:
    def __init__(self, w=None, alpha=None):
        if w is not None:
            assert isinstance(w, np.ndarray)
            assert w.ndim == 2
        assert isinstance(alpha, (int, float, np.number))
        self.w = w
        self.alpha = alpha

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if x_train.ndim == 1:
            x_train = x_train[:, None]
        if y_train.ndim == 1:
            y_train = y_train[:, None]
        self.w = np.linalg.inv(self.alpha * np.eye(x_train.shape[1]) + x_train.T @ x_train) @ x_train.T @ y_train

    def predict(self, x_test: np.ndarray):
        if x_test.ndim == 1:
            x_test = x_test[:, None]
        y = x_test @ self.w
        return y


class BayesianLinearRegression:
    """
    P(w) = N(w|0, alpha ^ -1 I)
    P(t|w) = N(t|Xw, beta ^ -1 I)
    p(w|t) = N(w|m,s)
    s = (alpha * I + beta * XTX)^-1
    m = beta * s * XT * t
    """
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None
        self.w_var = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.w_precision = self.alpha * np.eye(x_train.shape[1]) + self.beta * x_train.T @ x_train
        self.w_var = np.linalg.inv(self.w_precision)
        self.w_mean = self.beta * self.w_var @ x_train.T @ y_train

    def predict(self, x: np.ndarray, return_std=False):
        y = x @ self.w_mean
        if return_std:
            var = 1 / self.beta + np.sum(x @ self.w_var * x, axis=1)
            std = np.sqrt(var)
            return y, std
        return y

    def sample(self, x, size):
        w = np.random.multivariate_normal(self.w_mean, self.w_var, size)
        y = x @ w.T
        return y

# SGD
class LinearRegression:
    def __init__(self, w=None, beta=None):
        self.w = w
        self.beta = beta
        
    def fit(self, X, y, iters=20, learning_rate=1e-3):
        N, D = X.shape
        w = np.random.randn(D)
        index = np.arange(N)
        np.random.shuffle(index)
        for _ in range(iters):
            for i in index:
                temp = w.copy()
                w = w + learning_rate * (y[i] - X[i] @ w.T) * X[i]
                if np.allclose(temp, w):
                    break
        self.w = w
        beta = (y - X @ self.w).T @ (y - X @ self.w) / N
        self.beta = 1. / beta
        
    def predict(self, x):
        y = x @ self.w
        return y


if __name__ == '__main__':
    x_train ,y_train, x_test, y_test = load_data('./dataset/exp2.txt')
    rbf = RBF(np.linspace(0, 1, 50), 0.1)
    phi_train = rbf(x_train)
    phi_test = rbf(x_test)
    model = LinearRegression()
    model.fit(phi_train, y_train)
    y_pred = model.predict(phi_test)
    x = np.linspace(0, 1, 100)
    X = rbf(x)
    mse = MSE(y_pred, y_test)
    
    plt.scatter(x_train, y_train, c='blue', label='train_data')
    plt.scatter(x_test, y_test, c='orange', label='test_data')
    plt.plot(x, model.predict(X))

    plt.legend()
    plt.show()
    
    
    print(f"MSE:{mse}")

























