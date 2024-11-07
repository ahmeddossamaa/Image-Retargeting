import numpy as np
from cv2 import normalize, NORM_MINMAX, CV_8U

from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.SeamCarvingI import SeamCarvingI


class MiddleSCI(SeamCarvingI):
    _mid: int = 0

    _energy: np.array = None
    _matrix: np.array = None

    _new_image: np.array = None
    _new_matrix: np.array = None

    Matrix = None

    def __init__(self, img, ratio, converter=None, feature_map=None, prev_matrix: bool = False):
        self.prev_matrix = prev_matrix

        super(MiddleSCI, self).__init__(img, ratio, converter=converter, feature_map=feature_map)

    def get_matrix(self, new=False):
        return self._matrix if not new else self._new_matrix

    @Decorators.Loggers.log_class_method_time
    def accumulate(self, energy):
        width = self._width

        self._mid = self._height // 2

        self._energy = energy
        self._matrix = np.zeros((self._height, self._width))

        matrix = self._matrix

        def topdown(s, e, step):
            for j in range(s, e, step):
                for i in range(width):
                    if j == s:
                        matrix[j, i] = self._energy[j, i]
                    else:
                        matrix[j, i] = self._energy[j, i] + (
                            Helper.Image.forward_energy(matrix, self._energy, width, j, i, step)
                            if MiddleSCI.Matrix is None
                            else Helper.Image.forward_energy_3d(matrix, MiddleSCI.Matrix, self._energy, width, j, i, step)
                        )

        topdown(0, self._mid, 1)
        topdown(self._height - 1, self._mid - 1, -1)

        if self.prev_matrix:
            MiddleSCI.Matrix = self._matrix.copy()
            # MiddleSCI.Matrix = normalize(self._matrix.copy(), None, 0, 255, NORM_MINMAX, CV_8U)
        return self._matrix

    @Decorators.Loggers.log_class_method_time
    def remove(self, energy):
        # print("start removal")
        middle = self._matrix[self._mid - 1] + self._matrix[self._mid]

        self._new_image = [[] for _ in range(self._height)]
        self._new_matrix = [[] for _ in range(self._height)]

        image = self._image
        matrix = self._matrix

        def bottomup(i, s, e, step):

            for j in range(s, e, step):
                v, i = Helper.Math.get_min(matrix, j, i, 0, len(matrix[j]) - 1)

                self._new_image[j] = np.delete(image[j], i, axis=0)
                self._new_matrix[j] = np.delete(matrix[j], i, axis=0)

        for k in range(self._num_seams):
            # print(k)
            i = np.argmin(middle)

            middle = np.delete(middle, i, axis=0)

            bottomup(i, self._mid - 1, -1, -1)
            bottomup(i, self._mid, self._height, 1)

            image = self._new_image
            matrix = self._new_matrix

        return np.array(self._new_image)
