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

        index = -1
        for j in range(0, height):
            for i in range(0, width):
                v = min(
                    matrix[j - 1][max(i - 1, 0)],
                    matrix[j - 1][i],
                    matrix[j - 1][min(i + 1, len(matrix) - 1)]
                ) if j - 1 > 0 else 0

                v = v + energy[j, i]

                if v > 0:
                    index = j

                matrix[j][i] = v

        for k in range(w):
            new_image = []
            new_energy = []
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

                # new_energy.append(
                #     np.delete(energy[i], j, axis=0)
                # )

                new_matrix.append(
                    np.delete(matrix[i], j, axis=0)
                )

            # new_image.reverse()
            # new_energy.reverse()
            # new_matrix.reverse()

            image = new_image
            # energy = new_energy
            matrix = new_matrix

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
