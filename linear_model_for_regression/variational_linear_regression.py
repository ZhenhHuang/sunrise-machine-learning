import numpy as np
from scipy.special import gamma, digamma


class VariationalLinearRegression:
    def __init__(self, a0=None, b0=None, c0=None, d0=None, m0=None):
        self.a0 = a0 or 1.
        self.b0 = b0 or 1.
        self.c0 = c0 or 1.
        self.d0 = d0 or 1.
        self.m0 = m0

    def __init_params(self, X: np.ndarray):
        self.samples, self.ndim = X.shape
        self.mN = np.zeros(self.ndim) if self.m0 is None else self.m0
        self.aN = self.a0 + self.ndim * 0.5
        self.bN = self.b0
        self.cN = self.c0 + self.samples * 0.5
        self.dN = self.d0

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int):
        self.__init_params(X)
        self.ELBO = 0.
        for i in range(max_iter):
            E_alpha = self.aN / self.bN
            E_beta = self.cN / self.dN
            self.SN = np.linalg.inv(E_beta * X.T @ X + np.eye(self.ndim) * E_alpha)
            self.mN = E_beta * self.SN @ X.T @ y
            E_wTw = self.mN.T @ self.mN + np.trace(self.SN)
            self.bN = self.b0 + 0.5 * E_wTw
            self.dN = self.d0 + 0.5 * ((y - X@self.mN).T @ (y - X@self.mN) + np.trace(X.T@X@self.SN))
            ELBO = self._get_ELBO(X, y)
            if np.allclose(ELBO, self.ELBO):
                break
            self.ELBO = ELBO


    def _get_ELBO(self, X: np.ndarray, y: np.ndarray):
        item1 = self.samples * 0.5 * (digamma(self.cN) - np.log(self.dN) - np.log(2 * np.pi)) \
                - self.cN * ((y - X@self.mN).T @ (y - X@self.mN) + np.trace(X.T@X@self.SN))

        item2 = (self.c0 - 1) * (digamma(self.cN) - np.log(self.dN)) - self.d0 * self.cN / self.dN + self.c0 * np.log(self.d0) - np.log(gamma(self.c0))

        item3 = - self.ndim * 0.5 * np.log(2 * np.pi) + 0.5 * self.ndim * (digamma(self.aN) - np.log(self.bN)) - \
            self.aN * 0.5 * (self.mN.T @ self.mN + np.trace(self.SN)) / self.bN

        item4 = self.a0 * np.log(self.b0) + (self.a0 - 1) * (digamma(self.aN) - np.log(self.bN)) - self.b0 * self.aN / self.bN - np.log(gamma(self.a0))

        item5 = 0.5 * np.linalg.slogdet(self.SN)[1] + 0.5 * self.ndim * (1 + np.log(2 * np.pi))

        item6 = np.log(gamma(self.aN)) - (self.aN - 1) * digamma(self.aN) - np.log(self.bN) + self.aN

        item7 = (self.cN - 1) * digamma(self.cN) + np.log(self.dN) - self.cN - np.log(gamma(self.cN))

        return item1 + item2 + item3 + item4 + item5 + item6 + item7

    def predict(self, X: np.ndarray, return_std: bool = False):
        y = X @ self.mN
        if return_std:
            var = np.sum(X @ self.SN * X, axis=-1) + self.dN / self.cN
            return y, np.sqrt(var)
        return y




