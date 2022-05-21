import numpy as np
from linear_model_for_classification.label_encoder import one_hot_decoder


class KMeans:
    def __init__(self, classes: int = 2):
        self.classes = classes
        self.iter_means = []
        self.iter_classes = []

    def fit(self, X, iters=20, one_hot=False):
        n = X.shape[0]
        means = X[np.random.choice(np.arange(n), self.classes)]     # init randomly
        self.iter_means.append(means)
        for epoch in range(iters):
            R = np.zeros((n, self.classes))
            for i in range(n):
                # min_dist = np.inf
                # for k in range(self.classes):
                #     distance = self.dist(X[i], means[k])
                #     if distance <= min_dist:
                #         min_dist = distance
                #         index = k
                dist = self.dist(X[i], means)
                index = dist.argmax(axis=-1)
                R[i, index] = 1
            tmp = np.einsum('ij,ik->jk', R, X) / (np.sum(R, axis=0)[:, None] + 1e-4)
            # early stop
            if np.allclose(tmp, means):
                break
            means = tmp
            self.iter_means.append(means)
            self.iter_classes.append(R)
        self.means = means
        if one_hot:
            return R, means
        else:
            return one_hot_decoder(R), means

    def predict(self, x):
        dist = self.dist(x, self.means)
        return dist.argmax(axis=-1)

    def dist(self, x, y):
        if x.ndim == 1:
            x = x[None, :]
        return np.sum((x[:, None, :]-y[None, :, :])**2, axis=-1)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    X, y, true_centers = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0, return_centers=True)
    model = KMeans(classes=3)
    results, centers = model.fit(X, iters=20)