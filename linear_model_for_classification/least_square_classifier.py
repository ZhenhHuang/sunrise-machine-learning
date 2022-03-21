import numpy as np
from linear_model_for_classification.label_encoder import one_hot_encoder, one_hot_decoder


class LeastSquareClassifier:
    def __init__(self, w: np.ndarray = None):
        self.w = w

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs=3, lr=1e-4, use_SGD=False, batch_size=10):
        if y_train.ndim == 1:
            y_train = one_hot_encoder(y_train)
        if self.w is None:
            self.w = np.random.normal(size=(x_train.shape[1], y_train.shape[1]))
        if use_SGD:
            iteration = self.__getiter(batch_size, x_train, y_train)
            dataloader = self.__getdata(iteration)
            for epoch in range(epochs):
                np.random.shuffle(dataloader)
                for i, (x, y) in enumerate(dataloader):
                    x = x
                    y = y
                    grad_w = x.T @ (x @ self.w - y)
                    self.w = self.w - lr * grad_w
        else:
            self.w = np.linalg.pinv(x_train) @ y_train


    def classify(self, x):
        return np.argmax(x @ self.w, axis=-1)

    def __getiter(self, batch_size, x_train, y_train):
        datalist = list(zip(x_train, y_train))
        np.random.shuffle(datalist)
        sindex = 0
        eindex = batch_size
        while eindex <= len(datalist):
            batch = datalist[sindex: eindex]
            sindex = eindex
            eindex += batch_size
            yield batch

        if eindex > len(datalist) > sindex:
            batch = datalist[sindex:]
            yield batch

    def __getdata(self, iteration):
        dataset = []
        labelset = []
        for iter in iteration:
            data, label = [], []
            for x, y in iter:
                data.append(x[None, :])
                label.append(y[None, :])
            dataset.append(np.concatenate(data))
            labelset.append(np.concatenate(label))
        return list(zip(dataset, labelset))















