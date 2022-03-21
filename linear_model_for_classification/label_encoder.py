import numpy as np
"""
    One_hot encoding
    for instance,
    class = [0, 0, 1, 1, 2]
    one_hot = array([[1., 0., 0.],
                     [1., 0., 0.],
                     [0., 1., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])
"""


def one_hot_encoder(labels):
    labels = labels.astype(int)
    n_class = np.max(labels) + 1
    one_hot = np.eye(int(n_class))
    return one_hot[labels]


def one_hot_decoder(one_hot):
    classes = np.where(one_hot > 0)[1]
    return classes
