import numpy as np
from scipy.misc import imresize


def resize(digits, row_size, column_size):
    """
    Resize images from input scale to row-size x clumn_size
    @row_size,column_size : scale_size intended to be
    """

    return np.array([imresize(_, size=(row_size, column_size)) for _ in digits])