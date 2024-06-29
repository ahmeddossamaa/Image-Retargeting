import math
from threading import Thread

import numpy as np

from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.SeamCarving import SeamCarving
import concurrent.futures


class MiddleSC(SeamCarving):
    def __init__(self, img, energy, ratio, color=False):
        self._color = color
        super(MiddleSC, self).__init__(img, energy, ratio)

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img

        height, width, z = image.shape

        mid = height // 2

        w = int(width - width * self._ratio)

        matrix = np.zeros((height, width))

        new_image = []

        def accumulate(s, e, step):
            for j in range(s, e, step):
                for i in range(width):
                    if j == s:
                        matrix[j, i] = self._energy[j, i]
                    else:
                        left_bound = max(i - 1, 0)
                        right_bound = min(i + 1, width - 1)

                        cu_val = abs(self._energy[j - step, right_bound] - self._energy[j, left_bound])

                        cl_val = abs(self._energy[j - step, i] - self._energy[j, left_bound]) + cu_val
                        cr_val = abs(self._energy[j - step, i] - self._energy[j, right_bound]) + cu_val

                        matrix[j, i] = self._energy[j, i] + min(
                            cl_val + matrix[j - step, left_bound],
                            cu_val + matrix[j - step, i],
                            cr_val + matrix[j - step, right_bound],
                        )

        @Decorators.Loggers.log_method_time
        def execute():
            accumulate(0, mid, 1)
            accumulate(height - 1, mid - 1, -1)

        execute()

        new_image = [[] for i in range(height)]
        new_matrix = [[] for i in range(height)]

        middle = matrix[mid - 1] + matrix[mid]

        # @Decorators.Loggers.log_method_time
        def remove_seam(i, s, e, step):
            for j in range(s, e, step):
                v, i = Helper.Math.get_min(matrix, j, i, 0, len(matrix[j]) - 1)

                if self._color:
                    image[j][i] = [0, 255, 0]
                    new_image[j] = image[j]

                    matrix[j][i] = math.inf
                    new_matrix[j] = matrix[j]
                else:
                    new_image[j] = np.delete(image[j], i, axis=0)
                    new_matrix[j] = np.delete(matrix[j], i, axis=0)

        for k in range(w):
            i = np.argmin(middle)

            middle = np.delete(middle, i, axis=0)

            remove_seam(i, mid - 1, -1, -1)
            remove_seam(i, mid, height, 1)
            # t2.join()

            image = new_image
            matrix = new_matrix

        return np.array(new_image)
