import numpy as np
import operator
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, inputs, labels, train_size=420, valid_size=280, test_size=300):
        self.inputs = inputs
        self.labels = labels
        self.dataSet = self.normalize(self.inputs)
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size

    def normalize(self, inputs, Linear=True):
        X_normal = None
        '''
        Linear normalize
        '''
        if Linear:
            maxSet = np.amax(inputs, axis=0)
            minSet = np.amin(inputs, axis=0)
            X_normal = (inputs - minSet) / (maxSet - minSet)
        else:
            std = np.std(inputs, axis=0)
            mean = np.mean(inputs, axis=0)
            X_normal = (inputs - mean) / std

        # 返回归一化的数据集
        return X_normal

    def calc_dist(self, trainSet, vect):
        error = (vect - trainSet) ** 2
        dist = np.sum(error, axis=1) ** 0.5
        return dist

    def fit(self, x, train_set, train_label, k=3):
        Dist = self.calc_dist(train_set, x)
        SortedIdx = np.argsort(Dist)
        CountList = {}
        for i in range(k):
            CountList[train_label[SortedIdx[i]]] = CountList.get(train_label[SortedIdx[i]], 0) + 1
        SortedDict = sorted(CountList.items(), key=operator.itemgetter(1), reverse=True)
        return SortedDict[0][0]

    def ChooseK(self):
        Ks = np.arange(1, 40)
        dataSet = self.normalize(self.inputs)
        train_size = self.train_size
        valid_size = self.valid_size
        train_set, train_label = dataSet[:train_size, :], self.labels[:train_size]
        valid_set, valid_label = dataSet[train_size:train_size+valid_size, :], self.labels[train_size:train_size+valid_size]
        errorList = []
        for k in Ks:
            errorCount = 0
            for i in range(valid_size):
                result = self.fit(valid_set[i, :], train_set, train_label, k)
                if result != valid_label[i]:
                    errorCount += 1
            errorCount /= valid_size
            errorList.append(errorCount)

        fig = plt.figure(dpi=600)
        plt.plot(Ks, errorList)
        plt.xlabel('value of K')
        plt.ylabel('error_rate')
        plt.grid()
        plt.show()
        return np.argsort(np.array(errorList), kind='stable')[0] + 1, errorList

    def predict(self):
        test_size = self.test_size
        test_set, test_label = self.dataSet[-test_size:, :], self.labels[-test_size:]
        train_set, train_label = self.dataSet[:self.train_size, :], self.labels[:self.train_size]
        error = 0
        for i in range(test_size):
            result = self.fit(test_set[i, :], train_set, train_label, k=5)
            if result != test_label[i]:
                error += 1
        error_rate = error / test_size
        return 1-error_rate


def loadDataSet(path):
    dataSet = []
    label = []
    fr = open(path)
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataSet.append(list(map(float, line[:-1])))
        label.append(line[-1])
    return np.array(dataSet), label


if __name__ == '__main__':
    dataSet, label = loadDataSet('./data/datingTestSet.txt')
    knn = KNN(dataSet, label)
    best_k, errorList = knn.ChooseK()
    rate = knn.predict()
    print('accuracy:{:.3f}%'.format(rate*100))







