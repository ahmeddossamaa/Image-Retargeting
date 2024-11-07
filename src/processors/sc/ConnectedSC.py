import math

import numpy as np

from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.SeamCarving import SeamCarving


class ConnectedSC(SeamCarving):
    def __init__(self, img, energy, ratio, color=False):
        self._color = color
        super(ConnectedSC, self).__init__(img, energy, ratio)

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img

        energy = self._energy
        height, width, z = image.shape

        w = int(width - width * self._ratio)

        matrix = np.zeros((height, width))

        index = -1
        for j in range(0, height):
            for i in range(0, width):
                v = min(
                    matrix[j - 1][max(i - 1, 0)],
                    matrix[j - 1][i],
                    matrix[j - 1][min(i + 1, len(matrix[0]) - 1)]
                ) if j - 1 > 0 else 0

                v = v + energy[j, i]

                if v > 0:
                    index = j

                matrix[j][i] = v

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
            row = matrix[index]
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
