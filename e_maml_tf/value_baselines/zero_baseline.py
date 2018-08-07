import numpy as np
from .base import Baseline


class ZeroBaseline(Baseline):
    def get_param_values(self, **kwargs):
        return None

    def set_param_values(self, val, **kwargs):
        pass

    def fit(self, paths):
        pass

    def predict(self, path):
        return np.zeros_like(path["rewards"])
