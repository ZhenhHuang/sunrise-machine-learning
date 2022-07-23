import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_data(path):
    f = open(path)
    label = []
    data = []
    for line in f.readlines():
        line = line.strip().split('\t')
        label.append(int(line[-1])-1)
        data.append(list(map(float, line[:-1])))
    f.close()
    return np.array(data), np.array(label)


def split_data(data, label):
    train_size = int(data.shape[0] * 0.7)
    
    train_data = data[: train_size]
    train_label = label[: train_size]
    norm = StandardScaler()
    norm.fit(train_data)
    train_data = norm.transform(train_data)

    test_data = data[train_size:]
    test_label = label[train_size:]
    test_data = norm.transform(test_data)
    
    return train_data, train_label, test_data, test_label


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
 

if __name__ == '__main__':
    data, label = load_data('./dataset/exp1.txt')
    train_data, train_label, test_data, test_label = split_data(data, label)
    acc_list = []
    for k in range(1, 31):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data, train_label)
        y_pred, acc = knn.predict(test_data, test_label)
        acc_list.append(acc)
    plt.plot(list(range(1,31)), acc_list)
    plt.show()
    best = np.max(acc_list)
    best = np.where(acc_list == best)[0] + 1
    print(f'best Ks are {best}')