import numpy as np

from abc import abstractmethod

from config.decorators import Decorators
from util.Image import Image


class SeamCarvingI:
    def __call__(self, *args, **kwargs):
        return self.execute()

    def __init__(self,
                 image: Image,
                 ratio: float,
                 converter: type = None,
                 feature_map: np.array = None):

        self._origin = image
        self._image = image.rgb()
        self._height, self._width, self._z = self._image.shape

        if ratio < 0 or ratio >= 1:
            raise Exception("Invalid Ratio.")

        self._ratio = ratio

        self._converter = converter
        self._feature_map = feature_map

        self._num_seams = int(self._width - self._width * self._ratio)

    """
    Extract features from an image

    Input: Rgb Image
    Output: 2D Feature Map
    """

    def convert(self, origin):
        if self._feature_map is not None:
            return self._feature_map

        if not self._converter:
            return origin.gray()

        return self._converter(origin)().image()

    """
    Extract energy map by top down accumulation

    Input: Feature Map
    Output: Energy Map
    """

    @abstractmethod
    def accumulate(self, energy):
        return []

    """
    Removes Seams by bottom up to find the best seams

    Input: Rgb Image, Energy Map, Ratio
    Output: 2D Feature Map
    """

    @abstractmethod
    def remove(self, energy):
        return []

    """
    Executes the Algorithm
    """

    @Decorators.Loggers.log_class_method_time
    def execute(self):
        energy = self.convert(self._origin)
        energy = self.accumulate(energy)
        result = self.remove(energy)

        return np.array(result)
