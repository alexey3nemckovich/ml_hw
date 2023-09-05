import numpy as np


def str_to_ndarray(str_value: str):
    return np.fromstring(str_value.strip('[]'), sep=' ')
