import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BayesLogisticRegression:
    """
    We only consider a simple form,
    the prior p(w)=N(w|0, alpha^-1 * I)
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, max_iter: int = 100):
        self.w_map = np.random.normal(size=x_train.shape[1])
        self.w_prec = self.alpha * np.eye(x_train.shape[1])
        mean = np.zeros(x_train.shape[1])
        for i in range(max_iter):
            w_old = np.copy(self.w_map)
            y = sigmoid(x_train @ w_old)
            grad = (
                    x_train.T @ (y - y_train)
                    + self.w_prec @ (w_old - mean)
            )
            hessian = (x_train.T * y * (1 - y)) @ x_train + self.w_prec
            self.w_map -= np.linalg.solve(hessian, grad)
            if np.allclose(self.w_map, w_old):
                break
        self.w_prec = hessian

    def prob(self, x):
        mu_a = x @ self.w_map
        var_a = np.sum(np.linalg.solve(self.w_prec, x.T).T * x, axis=1)
        return sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))








