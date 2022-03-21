import numpy as np
from linear_model_for_classification.label_encoder import one_hot_encoder


def softmax(a):
    return np.exp(a) / np.sum(np.exp(a), axis=-1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, w: np.ndarray = None):
        self.w = w

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, max_iter: int = 1000, lr=1e-4):
        T = one_hot_encoder(y_train)
        if self.w is None:
            self.w = np.random.normal(size=(x_train.shape[1], T.shape[1]))

        for i in range(max_iter):
            w_old = self.w.copy()
            Y = softmax(x_train @ w_old)
            grad = x_train.T @ (Y - T)
            self.w = w_old - lr * grad
            if np.allclose(self.w, w_old):
                break

    def prob(self, x):
        return softmax(x @ self.w)

    def classify(self, x):
        return np.argmax(self.prob(x), axis=-1)

