import numpy as np
from cv2 import threshold, THRESH_BINARY

from config.decorators import Decorators
from config.plotter import Plotter
from util.Image import Image
from util.Processor import Processor
from cv2.saliency import StaticSaliencyFineGrained


class SaliencyMap(Processor):
    def __init__(self, image: Image, ts=None):
        self.ts = ts

        super(SaliencyMap, self).__init__(image)

    # @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        l_channel, a_channel, b_channel = Image.split(self._image.lab())

        Plotter.images([
            l_channel,
            a_channel,
            b_channel,
        ], 1, 3, off=True)

        _, saliency_map = StaticSaliencyFineGrained().create().computeSaliency(l_channel)

        if self.ts:
            _, saliency_map = threshold(saliency_map, self.ts, 1, THRESH_BINARY)

        self._image = np.array(saliency_map)
