import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class KNeighborsClassifier:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        self.labels = y
        self.data = X
        self.n_classes = y.max() + 1
        
    def predict(self, X, y):
        """_summary_
        X.shape: M, D
        self.data.shape: N, D
        """
        dist = (X[:, None, :] - self.data[None, :, :]) ** 2     # M, N, D
        dist = dist.sum(axis=-1) ** 0.5
        classes = []
        for i in range(X.shape[0]):
            index = np.argsort(dist[i])[: self.n_neighbors]
            label = self.labels[index]
            count = np.zeros(self.n_classes)
            for j in range(self.n_neighbors):
                count[label[j]] += 1
            classes.append(count.argmax())
        acc = (classes == y).sum() / y.shape[0]
        return np.array(classes), acc
