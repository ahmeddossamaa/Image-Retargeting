import math

import numpy as np

from config.decorators import Decorators
from config.plotter import Plotter
from src.processors.sc.SeamCarving import SeamCarving


class ForwardSCV2(SeamCarving):
    def __init__(self, img, energy, ratio, color=False):
        self._color = color
        super(ForwardSCV2, self).__init__(img, energy, ratio)

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img

        energy = self._energy
        height, width, z = image.shape

        w = int(width - width * self._ratio)

        matrix = np.zeros((height, width))

        def c(x, y):
            return d(image[y][x - 1], image[y][x + 1])

        def cl(x, y):
            if y == 0:
                return 0

            return c(x, y) + d(image[y - 1][x], image[y][x - 1])

        def cu(x, y):
            if y == 0:
                return d(image[y][x - 1], image[y][x + 1])

            return c(x, y)

        def cr(x, y):
            if y == 0:
                return 0

            return c(x, y) + d(image[y - 1][x], image[y][x + 1])

        def d(a, b):
            r1, g1, b1 = a
            r2, g2, b2 = b

            return abs(r1 - r2) + abs(b1 - b2) + abs(g1 - g2)

        def get_min(j, k, start, end):
            l = max(k - 1, start)
            r = min(k + 1, end)

            left = matrix[j - 1][l]
            mid = matrix[j - 1][k] + cu(k, j)
            right = matrix[j - 1][r]

            if l != start:
                left += cl(l, j)

            if r != end:
                right += cr(r, j)

            if left <= mid and left <= right:
                temp = left, l
            elif mid <= left and mid <= right:
                temp = mid, k
            else:
                temp = right, r

            # print(temp)

            return temp

        for j in range(height):
            for i in range(1, width - 1):
                s = 0
                e = len(matrix[0]) - 1

                v, t = get_min(j, i, s, e)

                matrix[j][i] = v

                pass

        Plotter.image(matrix, off=False)

        new_image = [[] for i in range(height)]
        new_matrix = [[] for i in range(height)]
        for k in range(w):
            row = matrix[0]
            i = np.argmin(row)

            for j in range(0, height, 1):
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
