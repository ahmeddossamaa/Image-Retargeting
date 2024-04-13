import numpy as np

from config.decorators import Decorators
from config.plotter import Plotter
from utils.Image import Image
from utils.Processor import Processor
from cv2.saliency import StaticSaliencyFineGrained


class SaliencyMap(Processor):
    @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        l_channel, a_channel, b_channel = Image.split(self._image.lab())

        Plotter.images([
            l_channel,
            a_channel,
            b_channel,
        ], 1, 3, off=True)

        _, saliency_map = StaticSaliencyFineGrained().create().computeSaliency(l_channel)

        self._image = np.array(saliency_map)
