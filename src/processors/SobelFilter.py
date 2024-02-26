import numpy as np

from config.constants import Filters
from config.decorators import Decorators
from utils.Processor import Processor
from scipy.signal import convolve2d


class SobelFilter(Processor):
    @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        sobel_filter = Filters.get('SOBEL')

        gx = np.abs(convolve2d(
            self._origin,
            sobel_filter.get('X'),
            mode='same'
        ))

        gy = np.abs(convolve2d(
            self._origin,
            sobel_filter.get('Y'),
            mode='same'
        ))

        self._image = np.sqrt(gx ** 2 + gy ** 2)
