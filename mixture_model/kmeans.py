import numpy as np
from linear_model_for_classification.label_encoder import one_hot_decoder


class KMeans:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.iter_means = []
        self.iter_result = []

    def fit(self, X, iters=20):
        n = X.shape[0]
        means = X[np.random.choice(n, self.n_classes)]
        for _ in range(iters):
            self.iter_means.append(means)
            self.means = means
            response = np.zeros((n, self.n_classes))
            dist = self.dist(X)
            for i in range(n):
                index = np.argmin(dist[i], axis=-1)
                response[i, index] = 1

            self.iter_result.append(response)
            means = np.einsum('nk,nd->kd', response, X) / (response.sum(0)[:, None] + 1e-4)
            if np.allclose(self.means, means):
                break
        return means, one_hot_decoder(response)

    def predict(self, x):
        dist = self.dist(x)
        return dist.argmin(-1)

    def dist(self, x):
        dist = ((x[:, None, :] - self.means[None, :, :]) ** 2).sum(-1)
        return dist


if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    X, y, true_centers = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0, return_centers=True)
    model = KMeans(classes=3)
    centers, results = model.fit(X, iters=20)