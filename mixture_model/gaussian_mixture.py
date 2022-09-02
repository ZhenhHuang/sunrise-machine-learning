import numpy as np
from linear_model_for_classification.label_encoder import one_hot_decoder, one_hot_encoder
from mixture_model.kmeans import KMeans
from scipy.stats import multivariate_normal


class GaussianMixture:
    def __init__(self, classes: int = 2):
        self.classes = classes
        self.iter_means = []
        self.iter_covs = []
        self.iter_coefs = []
        self.iter_classes = []

    def fit(self, X, iters: int = 20):
        means, covs, coefs = self._init_params(X, iters)
        self.means = means
        self.covs = covs
        self.coefs = coefs
        for epoch in range(iters):
            response = self._expectation(X)
            means, covs, coefs = self._maximum(X, response)
            result = np.argmax(response, axis=-1)
            self.iter_classes.append(result)
            if np.allclose(means, self.means) and np.allclose(covs, self.covs) and np.allclose(coefs, self.coefs):
                break
            self.iter_means.append(means)
            self.iter_covs.append(covs)
            self.iter_coefs.append(coefs)
            self.means = means
            self.covs = covs
            self.coefs = coefs
            self.response = response
        result = np.argmax(self.response, axis=-1)
        self.result = result

    def predict_prob(self, x):
        return self._expectation(x).max(-1)

    def predict(self, x):
        return self.predict_prob(x).argmax(-1)


    def _init_params(self, X, iters: int = 20):
        kmeans = KMeans(self.classes)
        means, latents = kmeans.fit(X, iters)
        classes = latents
        covs = []
        for i in range(self.classes):
            assert X.ndim == 2
            cov = np.cov(X[classes == i], rowvar=False)
            if X.shape[1] == 1:
                cov = cov[None, None, None]
            else:
                cov = cov[None, :, :]
            covs.append(cov)
        covs = np.concatenate(covs, axis=0)
        coefs = latents.sum(axis=0)
        return means, covs, coefs

    def __getpdf(self, X):
        pdfs = []
        for i in range(self.classes):
            pdf = multivariate_normal.pdf(X, self.means[i], self.covs[i])
            if pdf.ndim > 0:
                pdf = pdf[:, None]
            else:
                pdf = pdf[None, None]
            pdfs.append(pdf)
        pdfs = np.concatenate(pdfs, axis=-1)
        return pdfs

    def _expectation(self, X):
        response = self.coefs * self.__getpdf(X)
        response = response / (response.sum(axis=-1)[:, None] + 1e-4)
        return response

    def _maximum(self, X, response):
        N = response.sum(0)
        means = np.einsum('ij,ik->jk', response, X) / (N[:, None] + 1e-4)
        coefs = N / X.shape[0]
        d = X[None, :, :] - means[:, None, :]
        tmp = np.einsum('ij,jik->jik', response, d)
        covs = np.einsum('kij,kim->kjm', tmp, d) / (N[:, None, None] + 1e-4)
        return means, covs, coefs

    def _pdf(self, X):
        return (self.coefs * self.__getpdf(X)).sum(-1)

