import numpy as np

from config.decorators import Decorators
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from utils.Processor import Processor


class Combiner(Processor):
    @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        saliency = SaliencyMap(self._image, ts=0.3)().image()
        sobel = SobelFilter(self._image)().image()

        energy = np.maximum(sobel / 255, saliency)

        self._image = energy
