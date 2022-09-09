from collections import Counter
import numpy as np


def calc_Entropy(x):
    count = Counter([a.item() for a in x])
    probs = np.array([p / len(x) for p in count.values()])
    entropy = -np.sum(probs * np.log(probs))
    return entropy


def Calc_ConditonEntropy(x, y):
    """
    calculate -sum_x( p(x) sum_y( p(y|x)log(p(y|x)) ) )
    """
    count = Counter([a for a in x])
    p_x = np.array([p / len(x) for p in count.values()])
    E_y_x = []  # Entropy(Y|X=x)
    for feat in count.keys():
        sub_y = y[x == feat]
        E_y_x.append(calc_Entropy(sub_y))
    E_y_x = np.array(E_y_x)
    return np.sum(p_x * E_y_x)