import numpy as np

def _padding(x: np.ndarray, 
             padding: int = 0) -> np.ndarray:
    if padding > 0:
        x = np.pad(x, pad_width=padding, mode='constant', constant_values=0)
    return x

from .conv import Conv2D, Conv3D
from .maxpool import MaxPool2D, MaxPool3D