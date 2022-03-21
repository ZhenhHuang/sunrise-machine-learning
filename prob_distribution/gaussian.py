import numpy as np
from prob_distribution.random_variable import RandomVariable
from prob_distribution.gamma import Gamma


class Gaussian(RandomVariable):
    """
    Gaussian Distribution for single variable
    """

    def __init__(self, mean=None, var=None, prec=None):
        """
        :param mean: the mean of distribution
        :param var:  the variance of distribution
        :param prec: the precision of distribution
        """
        super(Gaussian, self).__init__()
        self.mean = None
        self.var = None
        self.prec = None
        if mean is not None:
            assert isinstance(mean, (float, np.number, Gaussian))
            self.mean = mean
            self.parameter["mean"] = mean

        if var is not None:
            assert isinstance(var, (float, np.number))
            self.var = var
            self.parameter["var"] = var
            self.prec = 1 / var
            self.parameter["prec"] = 1 / var

        if prec is not None:
            assert isinstance(prec, (float, np.number, Gamma))
            self.prec = prec
            self.parameter["prec"] = prec
            if not isinstance(prec, Gamma):
                self.var = 1 / prec
                self.parameter['var'] = 1 / prec

    def fit(self, x_train: np.ndarray):
        if isinstance(self.mean, Gaussian) or isinstance(self.prec, Gamma):
            self.bayes(x_train)
        else:
            self.ml(x_train)

    def ml(self, X):
        if X.ndim == 2:
            X = X.reshape(-1)
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        self.prec = 1. / self.var
        self.parameter['mean'] = self.mean
        self.parameter['var'] = self.var
        self.parameter['prec'] = self.prec

    def pdf(self, X):
        dis = X - self.mean
        return np.exp(-0.5 * self.prec * dis ** 2) / np.sqrt(2 * np.pi * self.var)

    def sample(self, amount: int):
        return np.random.normal(self.mean, np.sqrt(self.var), size=(amount,))

    def bayes(self, X: np.ndarray):
        N = len(X)
        if isinstance(self.mean, Gaussian):
            mu = np.mean(X, axis=0)
            mean = (self.var * self.mean.mean + N * self.mean.var * mu) / (N * self.mean.var + self.var)
            prec = self.mean.prec + N * self.prec
            self.mean = Gaussian(mean=mean, prec=prec)

        elif isinstance(self.prec, Gamma):
            a = self.prec.a + N / 2
            b = self.prec.b + N * np.var(X, axis=0) / 2
            self.prec = Gamma(a, b)
        else:
            raise TypeError






