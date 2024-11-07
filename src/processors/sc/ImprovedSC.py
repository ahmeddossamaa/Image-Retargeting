from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue

import numpy as np

from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.SeamCarvingI import SeamCarvingI


class ImprovedSC(SeamCarvingI):
    _mid: int = 0

    _energy: np.array = None
    _matrix: np.array = None

    Matrix = None

    def __init__(self, img, ratio, converter=None, feature_map=None, prev_matrix: bool = False):
        self.prev_matrix = prev_matrix

        super(ImprovedSC, self).__init__(img, ratio, converter=converter, feature_map=feature_map)

    def topdown(self, s, e, step):
        assert self._energy is not None, "Energy Map is None"

        height, width = self._energy.shape

        self._matrix = np.zeros((self._height, self._width))

        matrix = self._matrix

        for j in range(s, e, step):
            for i in range(width):
                if j == s:
                    self._matrix[j, i] = self._energy[j, i]
                else:
                    matrix[j, i] = self._energy[j, i] + (
                        Helper.Image.forward_energy(matrix, self._energy, width, j, i, step)
                        if ImprovedSC.Matrix is None
                        else Helper.Image.forward_energy_3d(matrix, ImprovedSC.Matrix, self._energy, width, j, i, step)
                    )

    @Decorators.Loggers.log_class_method_time
    def accumulate(self, energy):
        self._energy = energy

        self._mid = self._height // 2

        self._matrix = np.zeros((self._height, self._width))

        # with ThreadPoolExecutor() as executor:
        #     executor.submit(self.topdown, 0, self._mid, 1)
        #     executor.submit(self.topdown, self._height - 1, self._mid - 1, -1)

        self.topdown(0, self._height, 1)
        # self.topdown(self._height - 1, self._mid - 1, -1)

        if self.prev_matrix:
            ImprovedSC.Matrix = self._matrix.copy()

        return self._matrix

    def bottomup(self, indices, s, e, step):
        old_image = self._image
        old_matrix = self._matrix

        penalties = np.zeros((self._height, self._width))

        for k in range(self._num_seams):
            i = indices[k]

            for j in range(s, e, step):
                v, i = Helper.Math.get_min(old_matrix, j, i, 0, len(old_matrix[j]) - 1)

                if old_matrix[j, i] == np.inf:
                    penalties[j, i] += 1
                    continue

                old_matrix[j, i] = np.inf

        dh, dw = abs(e - s) + 1, abs(self._width - self._num_seams)

        # Plotter.images([old_matrix, self._energy], 1, 2)

        new_image = np.zeros((dh, dw, 3))
        for j in range(s, e, step):
            arr = []
            penalty = 0

            for i in range(self._width):
                if len(arr) == dw:
                    break

                if old_matrix[j, i] == np.inf:
                    penalty += penalties[j, i]
                    continue

                if penalty > 0:
                    penalty -= 1
                    continue

                arr.append(old_image[j, i])

            new_image[j % self._mid] = arr

        return new_image

    def remove_seam(self, indices, s, e, step):
        # print(s, e)

        old_image = self._image.copy()
        old_matrix = self._matrix.copy()

        new_image = [[] for i in range(self._height)]
        new_matrix = [[] for i in range(self._height)]

        for k in range(self._num_seams):
            # i = middle[k]
            i = np.argmin(indices)

            indices = np.delete(indices, i, axis=0)

            for j in range(s, e, step):
                v, i = Helper.Math.get_min(old_matrix, j, i, 0, len(old_matrix[j]) - 1)

                new_image[j] = np.delete(old_image[j], i, axis=0)
                new_matrix[j] = np.delete(old_matrix[j], i, axis=0)

            old_image = new_image.copy()
            old_matrix = new_matrix.copy()

        return np.array(new_image[s: e] if e > s else new_image[e + 1: s + 1])

    @Decorators.Loggers.log_class_method_time
    def remove(self, energy):
        dw = self._width - self._num_seams

        # middle = np.argsort(
        #     self._matrix[self._mid - 1] + self._matrix[self._mid]
        # )

        # middle = self._matrix[self._mid - 1] + self._matrix[self._mid]
        middle = self._matrix[-1]

        # with ThreadPoolExecutor() as exe:
        #     exe.submit(self.bottomup, middle, self._mid - 1, 0, -1)
        #     exe.submit(self.bottomup, middle, self._mid, self._height - 1, 1)

        self._image[:, :dw] = self.remove_seam(middle, self._height - 1, -1, -1)
        # self._image[self._mid:, :dw] = self.remove_seam(middle, 0, self._height, 1)

        return self._image[:, :dw]
