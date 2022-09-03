import numpy as np
from scipy.stats import norm
from prob_distribution.gaussian import Gaussian
from mixture_model.gaussian_mixture import GaussianMixture
from typing import Iterable


class FisherLinearDiscriminant:
    """
    Only for 2 classes
    """

    def __init__(self, w=None, threshold=None):
        self.w = w
        self.threshold = threshold

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        x0 = x_train[y_train == 0]
        x1 = x_train[y_train == 1]
        u1 = np.mean(x0, axis=0)
        u2 = np.mean(x1, axis=0)
        cov = np.cov(x0, rowvar=False) + np.cov(x1, rowvar=False)
        w = np.linalg.inv(cov) @ (u2 - u1)
        self.w = w / np.linalg.norm(w)
        g0 = Gaussian()
        g0.fit(x0 @ self.w)
        g1 = Gaussian()
        g1.fit(x1 @ self.w)
        x = np.roots([g1.var - g0.var,
                      2 * (g1.mean * g0.var - g0.mean * g1.var),
                      g1.var * g0.mean ** 2 - g0.var * g1.mean ** 2
                      - g1.var * g0.var * np.log(g1.var / g0.var)
                      ])
        if g0.mean < x[0] < g1.mean or g1.mean < x[0] < g0.mean:
            self.threshold = x[0]
        else:
            self.threshold = x[1]

    def project(self, x: np.ndarray):
        assert x.ndim <= 2, "ndim should be less than 3"
        if x.ndim == 1:
            x = x[None, :]
        return x @ self.w

    def classify(self, x: np.ndarray):
        assert x.ndim <= 2, "ndim should be less than 3"
        if x.ndim == 1:
            x = x[None, :]
        return (x @ self.w > self.threshold).astype(int)


class MultiFisherLinearDiscriminant:
    """
    For K >= 2
    """
    def __init__(self, W=None, threshold=None, n_classes=3, peaks: Iterable = None):
        self.W = W
        self.threshold = threshold
        self.n_classes = n_classes
        self.peaks = peaks or [2] * n_classes
        assert len(self.peaks) == self.n_classes, "peaks shape error"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        cov_b = []  # between
        cov_w = []  # within
        mean = []
        mu = x_train.mean(0, keepdims=True)  # 1 D
        for k in range(self.n_classes):
            x_k = x_train[y_train == k]  # N_k D
            mean_k = np.mean(x_k, axis=0, keepdims=True)  # 1 D
            mean.append(mean_k)
            dist = x_k - mean_k  # N_K D
            cov_k = dist.T @ dist
            cov_w.append(cov_k)
            dist = mean_k - mu
            cov_k = (y_train == k).sum() * dist.T @ dist
            cov_b.append(cov_k)
        self.mean = np.concatenate(mean, axis=0)
        cov_b = np.sum(cov_b, 0)  # D D
        cov_w = np.sum(cov_w, 0)
        A = np.linalg.inv(cov_w) @ cov_b
        _, vectors = np.linalg.eig(A)
        self.W = vectors[:, -(self.n_classes - 1):]  # D, K-1
        x_prj = x_train @ self.W
        self.__getDistributions(x_prj, y_train)

    def __getDistributions(self, x_train, y_train):
        distributions = []
        for k in range(self.n_classes):
            x_k = x_train[y_train == k]  # N_k D
            gmm = GaussianMixture(classes=self.peaks[k]) if self.peaks[k] > 1 else Gaussian()
            gmm.fit(x_k)
            distributions.append(gmm)
        self.distributions = distributions

    def project(self, x: np.ndarray):
        assert x.ndim <= 2, "ndim should be less than 3"
        if x.ndim == 1:
            x = x[None, :]
        return x @ self.W  # N, K-1

    def classify(self, x: np.ndarray):
        assert x.ndim <= 2, "ndim should be less than 3"
        probs = []
        if x.ndim == 1:
            x = x[None, :]
        x = self.project(x)
        for i in range(x.shape[0]):
            probs.append(np.concatenate([gmm._pdf(x[i]) for gmm in self.distributions]))
        classes = np.argmax(probs, axis=-1)
        return classes


if __name__ == '__main__':
    def create_data(size=50, add_outlier=False, add_class=False):
        assert size % 2 == 0
        x0 = np.random.normal(size=size).reshape(-1, 2) - 1
        x1 = np.random.normal(size=size).reshape(-1, 2) + 1
        if add_outlier:
            x = np.random.normal(size=10).reshape(-1, 2) + np.array([5, 10])
            return np.concatenate([x0, x1, x]), np.concatenate([np.zeros(size // 2), np.ones(size // 2 + 5)])
        if add_class:
            x = np.random.normal(size=size).reshape(-1, 2) + 3
            return np.concatenate([x0, x1, x]), np.concatenate(
                [np.zeros(size // 2), np.ones(size // 2), 2 * np.ones(size // 2)])
        return np.concatenate([x0, x1]), np.concatenate([np.zeros(size // 2), np.ones(size // 2)])


    model = MultiFisherLinearDiscriminant(n_classes=3)
    x_train, y_train = create_data(add_class=True)
    model.fit(x_train, y_train)
    import matplotlib.pyplot as plt
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    x_test = np.concatenate([x1_test, x2_test]).reshape(2, -1).T
    y_pred = model.classify(x_test)
    x = np.linspace(-5, 5, 20)
    plt.contourf(x1_test, x2_test, y_pred.reshape(100, -1), alpha=0.2, levels=np.linspace(0, 1, 3), cmap=plt.cm.RdBu_r)
    # plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
    # plt.plot(x, x * model.W[1] / model.W[0], label='w', linestyle='--')
    plt.title('Fisher Discriminant')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()

    # plt.plot(x, x * model.W[1] / model.W[0], label='w', linestyle='--')
    # w = model.W
    # rollmat = np.zeros((2, 2))
    # div = np.sqrt(w[0] ** 2 + w[1] ** 2)
    # rollmat[0, 0] = w[0] / div
    # rollmat[0, 1] = w[1] / div
    # rollmat[1, 0] = -w[1] / div
    # rollmat[1, 1] = w[0] / div
    # x_proj = x_train @ w
    # x_proj = np.concatenate([x_proj[:, None], np.zeros_like(x_proj[:, None])], axis=-1).reshape(-1, 2)
    # plt.scatter(x_proj[:,0], x_proj[:,1]-5, c=y_train)
    # x_roll = x_proj @ rollmat
    # plt.contourf(x1_test, x2_test, y_pred.reshape(100, -1), alpha=0.2, levels=np.linspace(0, 1, 3))
    # plt.scatter(x_roll[:, 0], x_roll[:, 1], c=y_train)
    # plt.scatter(0, 0, marker='x', alpha=1)
    # plt.title('Projection')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    # plt.legend()
    # plt.show()

