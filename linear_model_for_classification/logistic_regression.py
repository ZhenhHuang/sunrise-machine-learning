import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    """
    For 2 classes
    """
    def __init__(self, w: np.ndarray = None):
        self.w = w

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, max_iter: int = 100):
        if self.w is None:
            self.w = np.random.normal(size=x_train.shape[1])
        # t = one_hot_encoder(y_train)
        t = y_train
        for i in range(max_iter):
            w_old = self.w.copy()
            y = sigmoid(x_train @ self.w)
            R = np.diag(y*(1-y))
            H = x_train.T @ R @ x_train
            grad = x_train.T @ (y - t)
            self.w = w_old - np.linalg.pinv(H) @ grad
            if np.allclose(self.w, w_old):
                break

    def prob(self, x: np.ndarray):
        return sigmoid(x @ self.w)

    def classify(self, x: np.ndarray):
        return (sigmoid(x @ self.w) > 0.5).astype(int)






