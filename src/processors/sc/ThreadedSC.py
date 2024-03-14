from threading import Thread

import numpy as np

from config.decorators import Decorators
from config.helper import Helper
from src.processors.sc.SeamCarving import SeamCarving


class ThreadedSC(SeamCarving):
    def __int__(self, img, energy, ratio):
        super(ThreadedSC, self).__init__(img, energy, ratio)

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img
        energy = self._energy

        height, width, z = image.shape

        print("width", width)

        w = int(width - width * self._ratio)

        matrix = np.zeros((height, width))

        new_image = []

        index = -1

        @Decorators.Loggers.log_method_time
        def process_width(left, right):
            # print("--------------------")
            # print(left, right)
            for j in range(0, height):
                for i in range(left, right):
                    v = min(
                        matrix[j - 1][max(i - 1, 0)],
                        matrix[j - 1][i],
                        matrix[j - 1][min(i + 1, len(matrix) - 1)]
                    ) if j - 1 > 0 else 0

                    v = v + energy[j, i]

                    if v > 0:
                        index = j

                    matrix[j][i] = v

        temp = matrix

        @Decorators.Loggers.log_method_time
        def execute():
            n = 16
            dw = width // n
            for n in range(n):
                idx = n * dw

                t = Thread(target=process_width, args=(idx, idx + dw))

                t.start()

        execute()

        _, indices = Helper.Lists.sort_with_indices(matrix[index])

        for k in range(w):
            new_image = []
            new_matrix = []

            row = matrix[index]
            j = np.argmin(row)

            for i in range(height - 1, -1, -1):
                start = max(0, j - 1)
                end = min(j + 2, len(matrix[i - 1]) - 1)

                j = np.argmin(
                    matrix[i - 1][
                        start: end
                    ]
                ) + start

                new_image.append(
                    np.delete(image[i], j, axis=0)
                )

                new_matrix.append(
                    np.delete(matrix[i], j, axis=0)
                )

            image = new_image
            matrix = new_matrix

        return np.array(new_image), np.array(temp)