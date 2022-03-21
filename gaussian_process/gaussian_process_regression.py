import numpy as np


class GaussianProcessRegressor:
    """
    kernel: RBF(sigma_overall, l_scale)
    alpha: noise, 1-D array or scaler
    """
    def __init__(self, kernel, sigma_overall, l_scale, alpha=0.):
        self.kernel = kernel(sigma_overall, l_scale)
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.train_x_ = X
        self.train_y_ = y

    def predict(self, X, return_cov=True, return_std=False):
        if return_cov and return_std:
            raise RuntimeError("return_cov, return_std can't be True in the same time")
        if not hasattr(self, 'train_x_'):
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = self.kernel(X, X)
                return y_mean, y_cov
            elif return_std:
                y_cov = self.kernel(X, X)
                return y_mean, np.sqrt(np.diag(y_cov))
            else:
                return y_mean
        K = self.kernel(self.train_x_, self.train_x_)
        L = np.linalg.cholesky(K + self.alpha * np.eye(self.train_x_.shape[0]))
        alpha = np.linalg.solve(L, self.train_y_)
        alpha = np.linalg.solve(L.T, alpha)
        y_mean = self.kernel(self.train_x_, X).T @ alpha
        v = np.linalg.solve(L, self.kernel(self.train_x_, X))
        y_cov = self.kernel(X, X) - v.T @ v + self.alpha * np.eye(X.shape[0])
        if return_cov:
            return y_mean, y_cov
        elif return_std:
            return y_mean, np.sqrt(np.diag(y_cov))
        else:
            return y_mean

    def sample_func(self, X, n_samples=1):
        y_mean, y_cov = self.predict(X, return_cov=True, return_std=False)
        sampled_y = np.random.multivariate_normal(y_mean, y_cov, size=n_samples)
        return sampled_y








