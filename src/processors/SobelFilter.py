import numpy as np

from cv2 import normalize, CV_64F, Sobel, NORM_MINMAX, CV_8U, GaussianBlur
from config.decorators import Decorators
from util.Image import Image
from util.Processor import Processor


class SobelFilter(Processor):
    def __init__(self, image: Image, blur=False, ksize=3):
        self._blur = blur
        self._ksize = ksize
        super(SobelFilter, self).__init__(image.gray())

    # @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        if self._blur:
            self._origin = GaussianBlur(self._origin, (9, 9), 0)

        gradient_x = Sobel(self._origin, CV_64F, 1, 0, ksize=self._ksize)
        gradient_y = Sobel(self._origin, CV_64F, 0, 1, ksize=self._ksize)

        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        gradient_normalized = normalize(gradient_magnitude, None, 0, 255, NORM_MINMAX, CV_8U)

        self._image = gradient_normalized
