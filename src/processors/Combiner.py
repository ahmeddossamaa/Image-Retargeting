import numpy as np

from config.decorators import Decorators
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from utils.Image import Image
from utils.Processor import Processor


class Combiner(Processor):
    def __init__(self, image: Image, depth=None):
        self._depth = depth

        super(Combiner, self).__init__(image)

    @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        saliency = self._depth / 255 if self._depth is not None else SaliencyMap(self._image, ts=0.3)().image()
        sobel = SobelFilter(self._image)().image()

        energy = np.maximum(sobel / 255, saliency)

        self._image = energy
