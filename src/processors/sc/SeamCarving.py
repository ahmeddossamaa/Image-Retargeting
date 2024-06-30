from abc import abstractmethod
from util.Algorithm import Algorithm


class SeamCarving(Algorithm):
    def __init__(self, img, energy, ratio):
        super(SeamCarving, self).__init__()

        self._img = img
        self._energy = energy
        self._ratio = ratio

        self._height, self._width, self._z = self._img.shape

    @abstractmethod
    def _main(self, *args, **kwargs):
        pass
