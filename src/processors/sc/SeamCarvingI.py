import numpy as np

from abc import abstractmethod

from config.decorators import Decorators
from utils.Image import Image


class SeamCarvingI:
    def __call__(self, *args, **kwargs):
        return self.execute()

    def __init__(self,
                 image: Image,
                 ratio: float,
                 converter: type = None):

        self._origin = image
        self._image = image.rgb()
        self._height, self._width, self._z = self._image.shape

        if ratio < 0 or ratio >= 1:
            raise Exception("Invalid Ratio.")

        self._ratio = ratio

        self._converter = converter

        self._num_seams = int(self._width - self._width * self._ratio)

    """
    Extract features from an image

    Input: Rgb Image
    Output: 2D Feature Map
    """

    def _convert(self):
        if not self._converter:
            return self._origin.gray()

        return self._converter(self._origin)().image()

    """
    Extract energy map by top down accumulation

    Input: Feature Map
    Output: Energy Map
    """

    @abstractmethod
    def _accumulate(self, energy):
        return []

    """
    Removes Seams by bottom up to find the best seams

    Input: Rgb Image, Energy Map, Ratio
    Output: 2D Feature Map
    """

    @abstractmethod
    def _remove(self, energy):
        return []

    """
    Executes the Algorithm
    """

    @Decorators.Loggers.log_class_method_time
    def execute(self):
        energy = self._convert()
        energy = self._accumulate(energy)
        result = self._remove(energy)

        return np.array(result)
