import numpy as np

from prob_distribution.random_variable import RandomVariable
from scipy.special import gamma


class Gamma(RandomVariable):
    def __init__(self, a, b):
        super(Gamma, self).__init__()
        self.a = a
        self.b = b

    def pdf(self, X):
        return self.b ** self.a * X ** (self.a-1) * np.exp(-self.b * X) / gamma(self.a)

    def sample(self, amount=100):
        return np.random.gamma(self.a, 1 / self.b, size=(amount,))




