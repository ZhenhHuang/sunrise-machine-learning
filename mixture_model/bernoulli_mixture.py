import numpy as np
from prob_distribution.bernoulli import MultiBernoulli
from scipy.special import logsumexp


class BernolliMixture:
    def __init__(self, classes):
        self.classes = classes
        self.probs = []
        self.coeffs = []
        self.results = []

    def fit(self, X, iters=10, low=0.25, high=0.75):
        D = X.shape[-1]
        u, coeff = self._init_param(D, low, high)
        for epoch in range(iters):
            response = self._expectation(X, u, coeff)
            results = np.argmax(response, axis=-1)
            self.results.append(results)
            u, coeff = self._maximum(X, response)
            if np.allclose(self.coeffs[-1], coeff):
                break
            self.probs.append(u)
            self.coeffs.append(coeff)

        self.u = u
        self.coeff = coeff

    def _init_param(self, D, low, high):
        u = np.random.uniform(low, high, size=(self.classes, D))
        u = u / u.sum(1)[:, None]
        coeff = np.ones(self.classes) / self.classes
        self.probs.append(u)
        self.coeffs.append(coeff)
        return u, coeff

    def _expectation(self, X, u, coeff):
        p = np.log(coeff) + self.__getpmf(X, u)
        response = p - logsumexp(p, axis=-1)[:, None]
        response = np.exp(response)
        return response

    def _maximum(self, X, response):
        N_k = response.sum(0)
        coeffs = N_k / X.shape[0]
        u = np.einsum('nk,nd->kd', response, X) / N_k[:, None]
        return u, coeffs

    def __getpmf(self, X, u):
        pmfs = []
        for i in range(self.classes):
            mb = MultiBernoulli(u[i])
            pmfs.append(mb.log_pmf(X)[None, :])
        pmfs = np.concatenate(pmfs, axis=0)
        return pmfs.T

    def predict(self, x):
        return self._expectation(x, self.u, self.coeff).argmax(-1)

    def predict_prob(self, x):
        return self._expectation(x, self.u, self.coeff)


if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    from mixture_model import BernolliMixture
    import matplotlib.pyplot as plt

    images, targets = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
    classes = ['2', '3', '4']
    data = []
    label = [[i] * 200 for i in range(len(classes))]
    for l in classes:
        data.append(images[targets == l][: 200])
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    data = data.reshape(600, -1) / 255
    data[data >= 0.5] = 1
    data[data < 0.5] = 0
    bmm = BernolliMixture(classes=len(classes))
    bmm.fit(data)
