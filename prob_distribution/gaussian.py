import numpy as np
from prob_distribution.random_variable import RandomVariable
from prob_distribution.gamma import Gamma


class Gaussian(RandomVariable):
    """
    Gaussian Distribution for univariate variable
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
        self.ndim = 1
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

    def _pdf(self, X):
        return self.pdf(X)

    def pdf(self, X):
        dis = X - self.mean
        return np.exp(-0.5 * self.prec * dis ** 2) / np.sqrt(2 * np.pi * self.var)

    def sample(self, amount: int = 1):
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


class MultiGaussian(RandomVariable):

    def __init__(self, mean=None, cov: np.ndarray = None, prec: np.ndarray = None):
        super(MultiGaussian, self).__init__()
        self.mean = mean
        self.cov = cov
        self.prec = prec
        if self.mean is not None:
            if isinstance(self.mean, np.ndarray):
                self.parameter['mean'] = self.mean.flatten()
                self.ndim = mean.shape[-1]
            else:
                raise RuntimeError('mean is not ndarray')
        if self.cov is not None:
            if isinstance(self.cov, np.ndarray) and cov.ndim == 2:
                self.parameter['cov'] = self.cov
                self.parameter['prec'] = np.linalg.inv(self.cov)
                self.prec = self.parameter['prec']
            else:
                raise RuntimeError('input should be 2D-array')
        elif self.prec is not None:
            if isinstance(self.prec, np.ndarray):
                assert prec.ndim == 2, 'input should be 2D-array'
                self.parameter['prec'] = self.prec
                self.parameter['cov'] = np.linalg.inv(self.prec)
                self.cov = self.parameter['cov']
            else:
                raise RuntimeError('input should be 2D-array')

    def fit(self, X):
        if isinstance(X, np.ndarray):
            self.ndim = X.shape[-1]
            self.ml(X)

    def ml(self, X):
        assert X.ndim == 2
        self.mean = np.mean(X, axis=0)
        self.parameter['mean'] = self.mean
        self.cov = np.cov(X, rowvar=False)
        self.prec = np.linalg.inv(self.cov)
        self.parameter['cov'] = self.cov
        self.parameter['prec'] = self.prec

    def pdf(self, X):
        d = X - self.mean
        return (
                np.exp(-0.5 * np.sum(d @ self.prec * d, axis=-1))
                * np.sqrt(np.linalg.det(self.prec))
                / np.power(2 * np.pi, 0.5 * self.ndim))

    def sample(self, amount=1):
        return np.random.multivariate_normal(self.mean, self.cov, size=amount)

