from enum import Enum


# Setup
import numpy as np


class DataPath(Enum):
    INPUT_PATH = "../data/input"
    OUTPUT_PATH = "../data/output"


Filters = {
    'SOBEL': {
        'X': np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        'Y': np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
    }
}
