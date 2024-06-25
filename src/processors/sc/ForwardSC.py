import math

import numpy as np

from config.decorators import Decorators
from config.plotter import Plotter
from src.processors.sc.SeamCarving import SeamCarving


class ForwardSC(SeamCarving):
    def __init__(self, img, energy, ratio, color=False):
        self._color = color
        super(ForwardSC, self).__init__(img, energy, ratio)

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img

        energy = self._energy
        height, width, z = image.shape

        w = int(width - width * self._ratio)

        matrix = np.zeros((height, width))

        s = 0
        step = 1
        for j in range(0, height):
            for i in range(0, width):
                if j == s:
                    matrix[j, i] = self._energy[j, i]
                else:
                    left_bound = max(i - 1, 0)
                    right_bound = min(i + 1, width - 1)

                    cu_val = abs(self._energy[j - step, right_bound] - self._energy[j, left_bound])

                    cl_val = abs(self._energy[j - step, i] - self._energy[j, left_bound]) + cu_val
                    cr_val = abs(self._energy[j - step, i] - self._energy[j, right_bound]) + cu_val

                    # print(cl_val, cu_val, cr_val)

                    matrix[j, i] = self._energy[j, i] + min(
                        cl_val + matrix[j - step, left_bound],
                        cu_val + matrix[j - step, i],
                        cr_val + matrix[j - step, right_bound],
                    )

        Plotter.image(matrix, off=True)

        def get_min(j, k, start, end):
            l = max(k - 1, start)
            r = min(k + 1, end)

            left = matrix[j][l]
            mid = matrix[j][k]
            right = matrix[j][r]

            if left <= mid and left <= right:
                temp = left, l
            elif mid <= left and mid <= right:
                temp = mid, k
            else:
                temp = right, r

            return temp

        new_image = [[] for i in range(height)]
        new_matrix = [[] for i in range(height)]
        for k in range(w):
            row = matrix[0]
            i = np.argmin(row)

            for j in range(height - 1, -1, -1):
                start = 0
                end = len(matrix[j]) - 1

                v, i = get_min(j, i, start, end)

                if self._color:
                    image[j][i] = [0, 255, 0]
                    new_image[j] = image[j]

                    matrix[j][i] = math.inf
                    new_matrix[j] = matrix[j]
                else:
                    new_image[j] = np.delete(image[j], i, axis=0)

                    new_matrix[j] = np.delete(matrix[j], i, axis=0)

            image = new_image
            matrix = new_matrix

        return np.array(new_image)
