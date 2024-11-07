import numpy as np
from enum import Enum


class DataPath(Enum):
    INPUT_PATH = "../data/input"
    MODELS_PATH = "../data/models"
    OUTPUT_PATH = "../data/output"


class Events(Enum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    ERROR = "error"
    TEST = "test"


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

