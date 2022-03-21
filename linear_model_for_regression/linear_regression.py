import numpy as np

from linear_model_for_regression.regression import Regression


class LinearRegression(Regression):
    def __init__(self, w=None):
        if w is not None:
            assert isinstance(w, np.ndarray)
            assert w.ndim == 2
        self.w = w
        self.var = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if x_train.ndim == 1:
            x_train = x_train[:, None]
        if y_train.ndim == 1:
            y_train = y_train[:, None]
        self.w = np.linalg.pinv(x_train) @ y_train
        self.var = np.mean(np.square(y_train - x_train @ self.w))

    def predict(self, x_test: np.ndarray, return_std=False):
        if x_test.ndim == 1:
            x_test = x_test[:, None]
        y = x_test @ self.w
        if return_std:
            std = np.sqrt(self.var) + np.zeros_like(y)
            return y, std
        return y


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



























