import numpy as np

from config.decorators import Decorators
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from util.Image import Image
from util.Processor import Processor


class Combiner(Processor):
    def __init__(self, image: Image, depth=None, invert=False):
        self._depth = depth
        self._invert = invert

        super(Combiner, self).__init__(image)

    # @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        if self._depth is not None:
            if self._invert:
                self._depth = 255 - self._depth

            saliency = self._depth / 255
        else:
            saliency = SaliencyMap(self._image, ts=0.3)().image()
        sobel = SobelFilter(self._image)().image()

        energy = np.maximum(sobel / 255, saliency)

        self._image = energy
