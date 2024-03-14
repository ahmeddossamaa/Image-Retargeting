from abc import abstractmethod
from utils.Algorithm import Algorithm


class SeamCarving(Algorithm):
    def __init__(self, img, energy, ratio):
        super(SeamCarving, self).__init__()

        self._img = img
        self._energy = energy
        self._ratio = ratio

    @abstractmethod
    def _main(self, *args, **kwargs):
        pass
