import numpy as np
from typing import Union


def binary_encode(X: np.ndarray):
    """Binary encode

    Args:
        x (np.ndarray)
    """
    X = np.asarray(X)
    assert X.ndim <= 2, "dimension invalid"
    if X.ndim == 0:
        X = X[None, None]
    if X.ndim == 1:
        X = X[None, :]
    flag = isinstance(X[0, 0], (np.int32, np.int64, int))
    X = X.astype(str)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not flag:
                integer, decimal = X[i, j].split('.')
                integer = format(int(integer), 'b')
                decimal = format(int(decimal), 'b')
                X[i, j] = integer + "." + decimal
                print(X[i, j])
            else:
                integer = X[i, j]
                integer = format(int(integer), 'b')
                X[i, j] = integer
    return X, flag


def binary_decode(X: np.ndarray, flag=False):
    """Binary decode

    Args:
        X (np.ndarray): binary code
        flag: is int
    """
    X = np.asarray(X)
    assert X.ndim <= 2, "dimension invalid"
    if X.ndim == 0:
        X = X[None, None]
    if X.ndim == 1:
        X = X[None, :]
    prefix = "0b"
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not flag:
                integer, decimal = X[i, j].split('.')
                integer = str(int(prefix+integer, 2))
                decimal = str(int(prefix+decimal, 2))
                X[i, j] = float(integer + '.' + decimal)
            else:
                integer = X[i, j]
                integer = str(int(prefix+integer, 2))
                X[i, j] = int(integer)
    X = X.astype(np.int32) if flag else X.astype(np.float32)
    return X


if __name__ == '__main__':
    # X = np.array([[1.1], [2.2]])
    X = np.random.uniform(0, 2, (3,3))
    print(X)
    X, flag = binary_encode(X)
    print(X)
    X = binary_decode(X, flag)
    print(X)