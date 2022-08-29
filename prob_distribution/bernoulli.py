from prob_distribution.random_variable import RandomVariable
import numpy as np


class MultiBernoulli(RandomVariable):
    def __init__(self, u=None):
        self.u = u

    def log_pmf(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[None, :]
        p = self.u ** x * (1-self.u) ** (1-x)
        np.clip(p, 1e-10, 1 - 1e-10, out=p)
        p = np.sum(np.log(p), axis=-1)
        return p


