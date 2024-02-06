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
        # TODO: Implement Connected Seam Carving.
        pass

    @Decorators.log_class_method_time
    def __dis_connected(self, *args, **kwargs):
        image, energy, w, h = args

        height, width = image.shape

        matrix = np.zeros((height, width))

        new_image = [

        ]

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

            # remove pixel from image
            energy[j][index] = 600
            # temp = list(energy[j])
            # temp.pop(index)
            # new_image.append(temp)
            # np.delete(energy[j], index)
            # energy[j].pop(index)

        # print(len(new_image), len(new_image[0]))

        return energy
