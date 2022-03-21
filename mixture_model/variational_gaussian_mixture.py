import numpy as np
from scipy.special import digamma, gamma
from scipy.stats import t
import matplotlib.pyplot as plt


class VariationalGaussianMixture:
    def __init__(self, k_class, alpha0=None, m0=None, v0=None, beta0: float = 1., W0=1.):
        self.k_class = k_class
        self.alpha0 = alpha0 or 1 / k_class
        self.W0 = W0
        self.v0 = v0
        self.beta0 = beta0
        self.m0 = m0

    def __init_param__(self, X: np.ndarray):
        N, self.ndim = X.shape
        self.alpha0 = np.ones(self.k_class) + self.alpha0
        self.m0 = np.mean(X, axis=0) if self.m0 is None else np.zeros(self.ndim) + self.m0
        self.v0 = self.v0 or self.ndim
        self.N_k = N / self.k_class + np.zeros(self.k_class)
        index = np.random.choice(N, self.k_class, replace=False)
        self.beta = self.beta0 + self.N_k
        self.alpha = self.alpha0 + self.N_k
        self.mu = X[index]     # [K, self.ndim]
        self.v = self.v0 + self.N_k
        self.W0 = np.eye(self.ndim)
        self.W = np.tile(self.W0, (self.k_class, 1, 1))


    def _variational_expect(self, X: np.ndarray):
        delta = X[:, None, :] - self.mu    # [N, K, D]
        E_mu_lambda = self.ndim / self.beta + self.v * np.sum(
            np.einsum('kij,nkj->nki', self.W, delta) * delta, axis=-1)    # (N ,K)
        E_ln_pi = digamma(self.alpha) - digamma(self.alpha.sum())     # (K, )
        E_ln_det_lambda = digamma(0.5 * (self.v - np.arange(self.ndim)[:, None])).sum(axis=0) + self.ndim * np.log(2.) + np.linalg.slogdet(self.W)[1]   # (K, )
        rou = E_ln_pi + 0.5 * E_ln_det_lambda - self.ndim * 0.5 * np.log(2 * np.pi) - 0.5 * E_mu_lambda
        rou = np.exp(rou)
        r = rou / np.sum(rou, axis=-1)[:, None]
        return r

    def _variational_maximization(self, X: np.ndarray, r: np.ndarray):
        self.N_k = r.sum(0)
        self.xbar_k = (X.T @ r / self.N_k).T
        d = X[:, None, :] - self.xbar_k
        self.S_k = np.einsum('nki,nkj->kij', d, r[:, :, None]*d) / self.N_k[:, None, None]
        d = self.xbar_k - self.m0
        self.alpha = np.ones(self.k_class) / self.k_class + self.N_k
        self.beta = self.beta0 + self.N_k
        self.mu = (self.beta0 * self.m0 + self.N_k[:, None] * self.xbar_k) / self.beta[:, None]
        self.W = np.linalg.inv(
            np.linalg.inv(self.W0) + (self.N_k * self.S_k.T).T + (self.beta0 * self.N_k * np.einsum('ki,kj->kij', d, d).T / (self.beta0 + self.N_k)).T
        )
        self.v = self.v0 + self.N_k

    def fit(self, X, max_iter=10):
        self.__init_param__(X)
        for _ in range(max_iter):
            r = self._variational_expect(X)
            self._variational_maximization(X, r)

    def classify(self, X):
        return np.argmax(self._variational_expect(X), -1)

    def student_t(self, X):
        nu = self.v + 1 - self.ndim
        L = (nu * self.beta * self.W.T / (1 + self.beta)).T
        d = X[:, None, :] - self.mu
        maha_sq = np.sum(np.einsum('nki,kij->nkj', d, L) * d, axis=-1)
        return (
            gamma(0.5 * (nu + self.ndim))
            * np.sqrt(np.linalg.det(L))
            * (1 + maha_sq / nu) ** (-0.5 * (nu + self.ndim))
            / (gamma(0.5 * nu) * (nu * np.pi) ** (0.5 * self.ndim)))

    def pdf(self, X):
        return (self.alpha * self.student_t(X)).sum(axis=-1) / self.alpha.sum()


if __name__ == '__main__':
    x1 = np.random.normal(size=(100, 2))
    x1 += np.array([-5, -5])
    x2 = np.random.normal(size=(100, 2))
    x2 += np.array([5, -5])
    x3 = np.random.normal(size=(100, 2))
    x3 += np.array([0, 5])
    x_train = np.vstack((x1, x2, x3))

    x0, x1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    x = np.array([x0, x1]).reshape(2, -1).T

    vgmm = VariationalGaussianMixture(k_class=6)
    vgmm.fit(x_train)

    plt.scatter(x_train[:, 0], x_train[:, 1], c=vgmm.classify(x_train))
    plt.contour(x0, x1, vgmm.pdf(x).reshape(100, 100))
    plt.xlim(-10, 10, 100)
    plt.ylim(-10, 10, 100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()













