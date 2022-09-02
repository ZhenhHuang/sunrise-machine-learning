import numpy as np
from scipy.stats import norm
from prob_distribution.gaussian import Gaussian


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
                      2*(g1.mean*g0.var - g0.mean*g1.var),
                      g1.var * g0.mean ** 2 - g0.var * g1.mean ** 2
                      - g1.var * g0.var * np.log(g1.var / g0.var)
                      ])
        if g0.mean < x[0] < g1.mean or g1.mean < x[0] < g0.mean:
            self.threshold = x[0]
        else:
            self.threshold = x[1]

    def project(self, x: np.ndarray):
        return x @ self.w

    def classify(self, x: np.ndarray):
        return (x @ self.w > self.threshold).astype(int)


class MultiFisherLinearDiscriminant:
    def __init__(self, W=None, threshold=None, n_classes=3):
        self.W = W
        self.threshold = threshold
        self.n_classes = n_classes
        
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        cov_b = []  # between
        cov_w = []  # within
        mean = []
        mu = x_train.mean(0, keepdims=True) # 1 D
        for k in range(self.n_classes):
            x_k = x_train[y_train == k] # N_k D
            mean_k = np.mean(x_k, axis=0, keepdims=True)  # 1 D
            mean.append(mean_k)
            dist = x_k[:, None, :] - mean_k[:, :, None]  # N_K D D
            cov_k = np.einsum('nde,nde->ed', dist, dist)
            cov_w.append(cov_k)
            dist = mean_k - mu
            cov_k = (y_train == k).sum() * dist * dist.T
            cov_b.append(cov_k)
        cov_b = np.sum(cov_b, 0)    # D D 
        cov_w = np.sum(cov_w, 0)
        A = np.linalg.inv(cov_w) @ cov_w
        _, vectors = np.linalg.eig(A)
        self.W = vectors[:, -(self.n_classes-1):]
            





