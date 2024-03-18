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
    def __accumulate(self):
        index = -1

        matrix = self._matrix
        shift_map = np.zeros((self._height, self._width))

        start = 0
        end = len(matrix[0]) - 1

        def get_min(j, k, start, end):
            l = max(k - 1, start)
            r = min(k + 1, end)

            left = matrix[j][l]
            mid = matrix[j][k]
            right = matrix[j][r]

            if left <= mid and left <= right:
                return left, l
            elif mid <= left and mid <= right:
                return mid, k

            return right, r

        for j in range(0, self._height):
            for i in range(0, self._width):
                v = 0
                least_index = i
                if j - 1 > 0:
                    v, least_index = get_min(j - 1, i, start, end)

                    # v = matrix[j - 1][least_index]

                v = v + self._energy[j, i]

                if v > 0:
                    index = j

                matrix[j][i] = v
                shift_map[j][i] = int(least_index - i)

        return matrix, shift_map, index

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img

        height, width, z = image.shape

        w = int(width - width * self._ratio)

        new_image = []

        matrix, shift_map, index = self.__accumulate()

        # print(matrix)
        # print(shift_map)

        # Plotter.image(matrix)

        for k in range(w):
            new_image = []
            new_matrix = []

            row = matrix[index]
            i = np.argmin(row)

            for j in range(height - 1, -1, -1):
                start = max(0, i - 1)
                # end = min(i + 2, len(matrix[j - 1]) - 1)
                end = len(matrix[j - 1]) - 1

                # i = np.argmin(
                #     matrix[j - 1][
                #         start: end
                #     ]
                # ) + start

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

                i = min(int(i + shift_map[j][i]), end)

            image = new_image
            matrix = new_matrix

        return np.array(new_image)
