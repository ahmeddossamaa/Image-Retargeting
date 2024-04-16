import math
from threading import Thread

import numpy as np

from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.SeamCarving import SeamCarving


class ConnectedSCV3(SeamCarving):
    def __init__(self, img, energy, ratio, color=False):
        self._color = color
        super(ConnectedSCV3, self).__init__(img, energy, ratio)

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img
        energy = self._energy

        height, width, z = image.shape

        mid = height // 2

        w = int(width - width * self._ratio)

        matrix = np.zeros((height, width))

        new_image = []

        @Decorators.Loggers.log_method_time
        def accumulate(s, e, step):
            for j in range(s, e, step):
                for i in range(0, width):
                    idx = j - 1 if step == 1 else j + 1

                    b = idx > s if step == 1 else idx < s

                    v = min(
                        matrix[idx][max(i - 1, 0)],
                        matrix[idx][i],
                        matrix[idx][min(i + 1, len(matrix[0]) - 1)]
                    ) if b else 0

                    v = v + energy[j, i]

                    matrix[j][i] = v

        index = -1

        @Decorators.Loggers.log_method_time
        def execute():
            t1 = Thread(target=accumulate, args=(0, mid, 1))
            t2 = Thread(target=accumulate, args=(height - 1, mid - 1, -1))

            t1.start()
            t2.start()

        execute()

        middle = matrix[mid - 1] + matrix[mid]

        Plotter.image(matrix, off=True)

        for k in range(w):
            new_image = []
            new_matrix = []

            row = matrix[index]
            i = np.argmin(row)

            for j in range(height - 1, -1, -1):
                start = max(0, i - 1)
                end = min(i + 2, len(matrix[j]) - 1)

                i = np.argmin(
                    matrix[j][
                        start: end
                    ]
                ) + start

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

            image = new_image
            matrix = new_matrix

        return np.array(new_image)
