import numpy as np
from config.decorators import Decorators
from utils.Algorithm import Algorithm


class SeamCarving(Algorithm):
    def __init__(self, is_connected=False):
        super(SeamCarving, self).__init__()
        self.is_connected = is_connected

    def _main(self, *args, **kwargs):
        return \
            self.__dis_connected(*args, **kwargs) \
            if not self.is_connected \
            else self.__connected(*args, **kwargs)

    @Decorators.log_class_method_time
    def __connected(self, *args, **kwargs):
        image, energy, w, h = args

        height, width, z = image.shape

        matrix = np.zeros((height, width))

        # new_image = np.zeros((height, width - 1, z))
        new_image = []
        new_energy = []

        for j in range(0, height):
            min_v = np.inf
            index = 0
            for i in range(0, width):
                v = min(
                    matrix[j - 1][max(i - 1, 0)],
                    matrix[j - 1][i],
                    matrix[j - 1][min(i + 1, len(matrix) - 1)]
                ) if j - 1 > 0 else 0

                v = v + energy[j, i]

                if v <= min_v:
                    min_v = v
                    index = i

                matrix[j][i] = v

        row = matrix[-1]
        j = np.argmin(row)

        for k in range(w):
            new_image = []
            for i in range(height - 1, -1, -1):
                if j == 0:
                    j = np.argmin(matrix[i, j:j + 2])
                elif j == w - 1:
                    j = np.argmin(matrix[i, j - 1:j + 1]) + j - 1
                else:
                    j = np.argmin(matrix[i, j - 1:j + 2]) + j - 1

                matrix[i, j] = np.inf

                new_energy.append(
                    np.delete(energy[i], j, axis=0)
                )

                new_image.append(
                    np.delete(image[i], j, axis=0)
                )

            new_image.reverse()
            image = new_image

        return np.array(new_image), np.array(new_energy)

    @Decorators.log_class_method_time
    def __dis_connected(self, *args, **kwargs):
        image, energy, w, h = args

        height, width, z = image.shape

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

        return np.array(new_image), np.array(energy)
