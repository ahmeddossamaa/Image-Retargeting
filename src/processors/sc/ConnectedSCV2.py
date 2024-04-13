import math

import numpy as np

from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.SeamCarving import SeamCarving


class ConnectedSCV2(SeamCarving):
    def __init__(self, img, energy, ratio, color=False):
        super(ConnectedSCV2, self).__init__(img, energy, ratio)

        self._color = color

        self._matrix = np.zeros((self._height, self._width))

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img
        energy = self._energy

        height, width, z = image.shape

        seams = int(width - width * self._ratio)

        matrix = np.zeros((height, width))
        shift_map = np.zeros((height, width))

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

        new_image = []

        index = -1
        for j in range(0, height):
            for i in range(0, width):
                v, l = get_min(j - 1, i, 0, width - 1) if j - 1 > 0 else (0, i)

                v = v + energy[j, i]

                if v > 0:
                    index = j

                matrix[j][i] = v
                shift_map[j][i] = l - i

                # if np.abs(l - i) > 1:
                #     print(i, j, v, l - i)

        Plotter.image(matrix, off=True)

        for k in range(seams):
            new_image = []
            new_matrix = []

            row = matrix[index]
            i = np.argmin(row)

            for j in range(height - 1, -1, -1):
                # print(j, i, shift_map[j, i])

                if self._color:
                    image[j][i] = [0, 255, 0]
                    new_image.append(
                        image[j]
                    )

                    matrix[j][i] = math.inf
                    new_matrix.append(
                        matrix[j]
                    )
                else:
                    new_image.append(
                        np.delete(image[j], i, axis=0)
                    )

                    new_matrix.append(
                        np.delete(matrix[j], i, axis=0)
                    )

                i += int(shift_map[j][i])

            # raise Exception("test")

            image = new_image
            matrix = new_matrix

        return np.array(new_image)
