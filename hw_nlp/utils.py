import numpy as np


def str_to_ndarray(str_value: str) -> np.ndarray:
    """
    summarize_texts converts string to np.ndarrary and returns it.

    :param str_value: np.ndarrary string representation.
    :returns: np.ndarrary object.

    Example usage:
    str_to_ndarray(str_value='[0.24, 0.33]')
    """
    return np.fromstring(str_value.strip('[]'), sep=' ')
