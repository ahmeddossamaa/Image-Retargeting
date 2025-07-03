import numpy as np

from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.MiddleSCI import MiddleSCI
from util.Image import Image


class MiddleForwardSC(MiddleSCI):
    def __init__(self, img, ratio, converter=None, feature_map=None, prev_matrix: bool = False):
        super(MiddleForwardSC, self).__init__(img, ratio, converter=converter, feature_map=feature_map, prev_matrix=prev_matrix)

    @Decorators.Loggers.log_class_method_time
    def execute(self):
        rate = int(self._num_seams / 5)
        for k in range(self._num_seams):
            if k % rate == 0:
                if self._matrix is not None:
                    self._origin = Image("", data=result)
                    # Plotter.image(self._origin.rgb())

                    self._height, self._width, self._z = result.shape

                energy = self.convert(self._origin)

                self.accumulate(energy)

                # Plotter.image(self._matrix)

                middle = self._matrix[self._mid - 1] + self._matrix[self._mid]

            i = np.argmin(middle)

            middle = np.delete(middle, i, axis=0)

            result = self.remove(energy, i)

        return np.array(result)

    # @Decorators.Loggers.log_class_method_time
    def remove(self, energy, i):
        self._new_image = [[] for _ in range(self._height)]
        self._new_matrix = [[] for _ in range(self._height)]

        # image = self._image
        # matrix = self._matrix

        def bottomup(i, s, e, step):
            for j in range(s, e, step):
                v, i = Helper.Math.get_min(self._matrix, j, i, 0, len(self._matrix[j]) - 1)

                self._new_image[j] = np.delete(self._image[j], i, axis=0)
                self._new_matrix[j] = np.delete(self._matrix[j], i, axis=0)

        bottomup(i, self._mid - 1, -1, -1)
        bottomup(i, self._mid, self._height, 1)

        self._image = self._new_image
        self._matrix = self._new_matrix

        return np.array(self._new_image)