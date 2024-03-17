import numpy as np

from config.decorators import Decorators
from src.processors.sc.SeamCarving import SeamCarving


class DisconnectedSC(SeamCarving):
    def __init__(self, img, energy, ratio):
        super(DisconnectedSC, self).__init__(img, energy, ratio)

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        image = self._img
        energy = self._energy

        height, width, z = image.shape

        w = int(width - width * self._ratio)

        new_image = []
        new_energy = []

        matrix = np.zeros((height, width))

        for j in range(0, height):
            for i in range(0, width):
                v = min(
                    matrix[j - 1][max(i - 1, 0)],
                    matrix[j - 1][i],
                    matrix[j - 1][min(i + 1, len(matrix) - 1)]
                ) if j - 1 > 0 else 0

                v = v + energy[j, i]

                matrix[j][i] = v

            indices = [index for index, value in sorted(enumerate(matrix[j]), key=lambda x: x[1])[:w]]

            new_energy.append(
                np.delete(energy[j], indices, axis=0)
            )

            new_image.append(
                np.delete(image[j], indices, axis=0)
            )

        return np.array(new_image)
