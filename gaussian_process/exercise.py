import numpy as np
from basis_function import RBF, Polynomial
from linear_regression import BayesianLinearRegression
from variational_linear_regression import VariationalLinearRegression
from gaussian_process.gaussian_process_regression import GaussianProcessRegressor
from gaussian_process.kernel import RBFKernel
import matplotlib.pyplot as plt
np.random.seed(100)


def load_data(path):
    """

    :param path: you need to use your path
    :return: data, label
    """
    f = open(path)
    dataset = []
    for line in f.readlines():
        line = line.strip().split()
        line = list(map(float, line))
        dataset.append(line)
    dataset = np.array(dataset)
    train = dataset[: 150]
    np.random.shuffle(train)
    test = dataset[150:]
    return train[:, 0], train[:, 1], test[:, 0], test[:, 1]


def main(path, alpha, beta):
    x_train, y_train, x_test, y_test = load_data(path)
    # rbf = RBF(np.linspace(0, 1, 50), 0.1)
    rbf = Polynomial(degree=3)
    X_train = rbf(x_train)
    X_test = rbf(x_test)
    model = BayesianLinearRegression(alpha, beta)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    x = np.linspace(0, 1, 100)
    X = rbf(x)
    plt.scatter(x_train, y_train, c='blue', label='train_data')
    plt.scatter(x_test, y_test, c='orange', label='test_data')
    plt.plot(x, model.predict(X))
    plt.show()
    test_error = np.square(y_pred - y_test).sum() / y_pred.shape[-1]
    return model, test_error


def variational_main(path):
    """

    :param path: path of data
    :return: model and error
    """
    x_train, y_train, x_test, y_test = load_data(path)
    rbf = RBF(np.linspace(0, 1, 50), 0.1)
    X_train = rbf(x_train)
    X_test = rbf(x_test)
    model = VariationalLinearRegression()
    model.fit(X_train, y_train, 50)
    y_pred = model.predict(X_test)
    x = np.linspace(0, 1, 100)
    X = rbf(x)
    plt.scatter(x_train, y_train, c='blue', label='train_data')
    plt.scatter(x_test, y_test, c='orange', label='test_data')
    plt.plot(x, model.predict(X))
    plt.show()
    test_error = np.square(y_pred - y_test).sum() / y_pred.shape[-1]
    return model, test_error


def gp_main(path):
    x_train, y_train, x_test, y_test = load_data(path)
    gp = GaussianProcessRegressor(RBFKernel, 1., 1e-2, 1e-4)
    gp.fit(x_train, y_train)
    y_mean, y_cov = gp.predict(x_test)
    test_error = np.square(y_mean - y_test).sum() / y_mean.shape[-1]
    x = np.linspace(0, 1, 100)
    y, y_cov = gp.predict(x)
    uncertain = 1.96 * np.sqrt(np.diag(y_cov))
    plt.scatter(x_train, y_train, c='blue', label='train_data')
    plt.scatter(x_test, y_test, c='orange', label='test_data')
    plt.fill_between(x, y - uncertain, y + uncertain, alpha=0.5, facecolor='yellow')
    plt.plot(x, y, label='predict')
    plt.legend()
    plt.savefig('gp_test.png')
    plt.show()
    return gp, test_error


if __name__ == '__main__':
    path = "../data/ex0.txt"
    # model, test_error = main(path, 0, 1)
    # model, test_error = variational_main(path)
    model, test_error = gp_main(path)
    print(f"test_error={test_error}")
