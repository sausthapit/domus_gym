import numpy as np


class MinMaxTransform:
    def __init__(self, min_value, max_value):
        assert max_value.shape == min_value.shape
        assert (max_value > min_value).all()

        self.min_value = min_value
        self.max_minus_min = max_value - min_value

    def transform(self, value):
        assert value.shape == self.min_value.shape
        return np.divide(value - self.min_value, self.max_minus_min)

    def inverse_transform(self, value):
        assert value.shape == self.min_value.shape
        # x = (v - min) / (max - min)
        # x * (max - min) = (v - min)
        # min + x * (max - min) = v
        return self.min_value + np.multiply(value, self.max_minus_min)
