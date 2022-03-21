import numpy as np


class RBFKernel:
    def __init__(self, sigma, scale):
        self.sigma = sigma
        self.scale = scale

    def __call__(self, x1: np.ndarray, x2: np.ndarray):
        m, n = x1.shape[0], x2.shape[0]
        K_matrix = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                K_matrix[i, j] = self.sigma * np.exp(-0.5 * np.sum((x1[i] - x2[j]) ** 2) / self.scale)
        return K_matrix

