import math

import numpy as np

from config.decorators import Decorators
from src.processors.sc.SeamCarving import SeamCarving


class DisconnectedSCV2(SeamCarving):
    def __init__(self, img, energy, ratio):
        super(DisconnectedSCV2, self).__init__(img, energy, ratio)

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img
        energy = self._energy

        height, width, z = image.shape

        # w = int(width - width * self._ratio)
        w = 1

        columns = 12

        dw = width // columns

        new_image = []

        indices = []
        values = []

        for j in range(0, height):
            sorted_list = sorted(enumerate(energy[j]), key=lambda x: x[1])[:2]

            if j == 0:
                indices = [index for index, value in sorted_list]
                # values = [value for index, value in sorted_list]
                continue

            temp_indices = []
            # temp_values = []

            for k in indices:
                # idx = k // dw
                #
                # start = max(0, k - dw // 2)
                # end = min(width - 1, k + dw // 2)

                # print(start, end)

                start = 0
                end = len(energy[j]) - 1

                least_index = 0

                least_value = min(
                    energy[j][max(k - 1, start)],
                    energy[j][k],
                    energy[j][min(k + 1, end)]
                )

                if least_value == energy[j][max(k - 1, start)]:
                    least_index = max(k - 1, start)
                elif least_value == energy[j][k]:
                    least_index = k
                elif least_value == energy[j][min(k + 1, end)]:
                    least_index = min(k + 1, end)

                if least_index in temp_indices:
                    print(k, indices)
                    print(least_index, temp_indices)
                    raise Exception("Duplicate")

                temp_indices.append(least_index)

                energy[j][least_index] = math.inf
                # temp_values.append(least_value)

                # dv = np.abs(v - val)
                # if dv > 100:
                #     pass

            indices = temp_indices
            # values = temp_values

            # row = np.delete(image[j], indices, axis=0)

            for i in indices:
                image[j][i] = [0, 255, 0]

            row = image[j]

            new_image.append(
                row
            )

        return np.array(new_image)
