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

    def topdown(self, s, e, step):
        assert self._energy is not None, "Energy Map is None"

        height, width = self._energy.shape

        for j in range(s, e, step):
            for i in range(width):
                if j == s:
                    self._matrix[j, i] = self._energy[j, i]
                else:
                    left_bound = max(i - 1, 0)
                    right_bound = min(i + 1, width - 1)

                    cu_val = abs(self._energy[j - step, right_bound] - self._energy[j, left_bound])

                    cl_val = abs(self._energy[j - step, i] - self._energy[j, left_bound]) + cu_val
                    cr_val = abs(self._energy[j - step, i] - self._energy[j, right_bound]) + cu_val

                    # print(cl_val, cu_val, cr_val)

                    self._matrix[j, i] = self._energy[j, i] + min(
                        cl_val + self._matrix[j - step, left_bound],
                        cu_val + self._matrix[j - step, i],
                        cr_val + self._matrix[j - step, right_bound],
                    )

    @Decorators.Loggers.log_class_method_time
    def _accumulate(self, energy):
        self._energy = energy

        self._mid = self._height // 2

        self._matrix = np.zeros((self._height, self._width))

        # with ThreadPoolExecutor() as executor:
        #     executor.submit(self.topdown, 0, self._mid, 1)
        #     executor.submit(self.topdown, self._height - 1, self._mid - 1, -1)

        self.topdown(0, self._mid, 1)
        self.topdown(self._height - 1, self._mid - 1, -1)

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

    def remove_seam(self, middle, s, e, step):
        old_image = self._image
        old_matrix = self._matrix

        new_image = [[] for i in range(self._height)]
        new_matrix = [[] for i in range(self._height)]

        for k in range(self._num_seams):
            i = middle[k]

            for j in range(s, e, step):
                v, i = Helper.Math.get_min(old_matrix, j, i, 0, len(old_matrix[j]) - 1)

                new_image[j] = np.delete(old_image[j], i, axis=0)
                new_matrix[j] = np.delete(old_matrix[j], i, axis=0)

            old_image = new_image
            old_matrix = new_matrix

        return np.array(new_image[s: e] if e > s else new_image[e: s])

    @Decorators.Loggers.log_class_method_time
    def _remove(self, energy):
        dw = self._width - self._num_seams

        middle = np.argsort(
            self._matrix[self._mid - 1] + self._matrix[self._mid]
        )

        # with ThreadPoolExecutor() as exe:
        #     exe.submit(self.bottomup, middle, self._mid - 1, 0, -1)
        #     exe.submit(self.bottomup, middle, self._mid, self._height - 1, 1)

        self._image[:self._mid, :dw] = self.bottomup(middle, self._mid - 1, 0, -1)
        self._image[self._mid:, :dw] = self.bottomup(middle, self._mid, self._height - 1, 1)

        return self._image[:, :dw]
