import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class Polynomial:
    def __init__(self, degree: int):
        self.degree = degree

    def __call__(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[:, None]
        p = PolynomialFeatures(self.degree)
        return p.fit_transform(x)


class RBF:
    def __init__(self, mean: np.ndarray, var):
        if mean.ndim == 1:
            mean = mean[:, None]
        self.mean = mean
        assert var > 0
        self.var = var

    def __call__(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[:, None]
        basis = [np.ones(len(x))]
        for mean in self.mean:
            delta = -np.sum((x - mean) ** 2, axis=-1) / (2 * self.var)
            basis.append(np.exp(delta))
        return np.array(basis).T


class Sigmoid:
    def __init__(self, mean: np.ndarray, std):
        if mean.ndim == 1:
            mean = mean[:, None]
        self.mean = mean
        assert std > 0
        self.std = std

    def __call__(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[:, None]
        basis = [np.ones(len(x))]
        for mean in self.mean:
            a = (x - mean) / self.std
            a = 1 / (1 + np.exp(-a))
            basis.append(a.flatten())
        return np.array(basis).T
