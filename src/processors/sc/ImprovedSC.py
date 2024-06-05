from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue

import numpy as np

from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.SeamCarvingI import SeamCarvingI
import cupy as cp


class ImprovedSC(SeamCarvingI):
    _mid: int = 0

    _energy: np.array = None
    _matrix: np.array = None

    @Decorators.Loggers.log_class_method_time
    def _accumulate(self, energy):
        self._energy = energy

        self._mid = self._height // 2

        self._matrix = np.zeros((self._height, self._width))

        def topdown(s, e, step):
            assert self._energy is not None, "Energy Map is None"

            for j in range(s, e, step):
                for i in range(0, self._width):
                    idx = j - 1 if step == 1 else j + 1

                    b = idx > s if step == 1 else idx < s

                    v = min(
                        self._matrix[idx][max(i - 1, 0)],
                        self._matrix[idx][i],
                        self._matrix[idx][min(i + 1, len(self._matrix[0]) - 1)]
                    ) if b else 0

                    v = v + self._energy[j, i]

                    self._matrix[j][i] = v

        with ThreadPoolExecutor() as executor:
            executor.submit(topdown, 0, self._mid, 1)
            executor.submit(topdown, self._height - 1, self._mid - 1, -1)

        # topdown(0, self._mid, 1)
        # topdown(self._height - 1, self._mid - 1, -1)

        # Plotter.image(self._matrix)

        return self._matrix

    def bottomup(self, indices, s, e, step):
        old_image = self._image
        old_matrix = self._matrix

        penalties = np.zeros((self._width, self._height))

        for k in range(self._num_seams):
            i = indices[k]

            for j in range(s, e, step):
                v, i = Helper.Math.get_min(old_matrix, j, i, 0, len(old_matrix[j]) - 1)

                if old_matrix[j, i] == np.inf:
                    penalties[j, i] += 1

                old_matrix[j, i] = np.inf

        dh, dw = abs(e - s), abs(self._width - self._num_seams)

        new_image = np.zeros((dh, dw, 3))
        for j in range(s, e, step):
            arr = []
            penalty = 0
            for i in range(self._width):
                if old_matrix[j, i] == np.inf:
                    penalty += penalties[j, i]
                    continue

                if penalty > 0:
                    penalty -= 1
                    continue

                arr.append(old_image[j, i])

            new_image[j % self._mid] = arr

        return new_image

    @Decorators.Loggers.log_class_method_time
    def _remove(self, energy):
        dw = self._width - self._num_seams

        middle = np.argsort(
            self._matrix[self._mid - 1] + self._matrix[self._mid]
        )

        # with ProcessPoolExecutor() as exe:
        #     left = exe.submit(self.bottomup, middle, 0, self._mid, 1)
        #     right = exe.submit(self.bottomup, middle, self._height - 1, self._mid - 1, -1)
        #
        #     while not left.done() or not right.done():
        #         pass
        #
        #     self._image[:self._mid, :dw] = left.result()
        #     self._image[self._mid:, :dw] = right.result()

        self._image[:self._mid, :dw] = self.bottomup(middle, 0, self._mid, 1)
        self._image[self._mid:, :dw] = self.bottomup(middle, self._height - 1, self._mid - 1, -1)

        return self._image[:, :dw]
