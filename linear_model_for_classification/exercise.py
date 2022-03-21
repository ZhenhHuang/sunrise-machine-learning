import numpy as np
import os
from linear_model_for_classification import FisherLinearDiscriminant, SoftmaxRegression
from linear_model_for_classification.softmax_regression import softmax
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from linear_model_for_regression import Polynomial


feature = Polynomial(degree=1)


def load_data(root_path, data_path):
    """

    :param root_path: root path of data
    :param data_path: name of data
    :return: datas, labels: arrays which are normalized
             label_dict
    """
    path = os.path.join(root_path, data_path)
    f = open(path, 'r')
    label_dict = {}
    labels = []
    datas = []
    for line in f.readlines():
        line = line.strip().split('\t')
        if line[-1] not in label_dict:
            label_dict[line[-1]] = len(label_dict)
        labels.append(int(label_dict[line[-1]]))
        data = list(map(float, line[: -1]))
        datas.append(data)
    datas = np.array(datas)
    scaler = StandardScaler()
    scaler.fit(datas)
    return scaler.transform(datas), np.array(labels), label_dict


def train(train_data, train_label):
    print('****training****')
    model = SoftmaxRegression()
    model.fit(train_data, train_label)
    return model


def val(model, val_data, val_label):
    print('****valid****')
    result = model.classify(val_data)
    result = result.flatten()
    result = (result == val_label).astype(int)
    acc = result.sum() / len(result)
    print(f'accuracy of valid: {acc * 100}%')


def test(model, test_data, test_label):
    print('****testing****')
    result = model.classify(test_data)
    result = result.flatten()
    result = (result == test_label).astype(int)
    acc = result.sum() / len(result)
    print(f'accuracy of test set: {acc * 100}%')
    return result


def split_train_test(datas, labels):
    num = len(labels)
    train_num = int(0.7 * num)
    val_num = int(0.1 * num)
    train_data, train_label = datas[:train_num], labels[:train_num]
    val_data, val_label = datas[train_num: train_num+val_num], labels[train_num: train_num+val_num]
    test_data, test_label = datas[train_num + val_num:], labels[train_num + val_num:]
    return train_data, train_label, val_data, val_label, test_data, test_label


def visualize(test_data, test_label, result):
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label)
    for i, r in enumerate(result):
        if r == 0:
            plt.scatter(test_data[i, 0], test_data[i, 1], c='none', marker='o', edgecolors='r', alpha=1)
    plt.gca().set_aspect('equal')
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.show()


def main(root_path, data_path):
    data, label, label_dict = load_data(root_path, data_path)
    train_data, train_label, val_data, val_label, test_data, test_label = split_train_test(data, label)
    model = train(feature(train_data), train_label)
    val(model, feature(val_data), val_label)
    result = test(model, feature(test_data), test_label)
    visualize(test_data, test_label, result)
    return model, result


if __name__ == '__main__':
    model, result = main('../data', 'datingTestSet.txt')








