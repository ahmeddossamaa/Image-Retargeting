import numpy as np

from config.decorators import Decorators
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from src.processors.preprocessing.Midas import MiDaS_predict
from util.Image import Image
from util.Processor import Processor


class Midas(Processor):
    def __init__(self, image: Image, depth=None, invert=False):
        self._depth = depth
        self._invert = invert

        super(Midas, self).__init__(image)

    @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        self._image = MiDaS_predict(self._origin) / 255
