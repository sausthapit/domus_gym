import numpy as np


class MinMaxTransform:
    def __init__(self, min_value, max_value):
        assert max_value.shape == min_value.shape
        assert (max_value > min_value).all()

        self.min_value = min_value
        self.max_minus_min_on_2 = (max_value - min_value) / 2

    def transform(self, value):
        assert value.shape == self.min_value.shape
        return np.divide(value - self.min_value, self.max_minus_min_on_2) - 1

    def inverse_transform(self, value):
        assert value.shape == self.min_value.shape
        # x = 2 * (v - min) / (max - min) - 1
        # (x + 1) * (max - min) / 2 = (v - min)
        # min + (x + 1) * (max - min) / 2 = v
        return self.min_value + np.multiply(value + 1, self.max_minus_min_on_2)
