import numpy as np
from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_UNCHANGED, imwrite, imdecode, IMREAD_COLOR, COLOR_BGR2RGB, cvtColor, \
    COLOR_BGR2GRAY, COLOR_RGB2BGR

from config.constants import DataPath
from config.decorators import Decorators
from config.helper import Helper


class Image:
    def __init__(self, file, gray=True, decode=False):
        self.__file = file
        self.__gray = gray

        self.__img = self.__read() if not decode else self.decode()

    def __call__(self, *args, **kwargs):
        return self.__img.copy()

    @Decorators.Loggers.log_class_method_time
    def __read(self):
        try:
            return imread(
                self.__file,
                Image._get_color(self.__gray)
            )
        except Exception as e:
            print(e)
            return None

    @Decorators.Loggers.log_class_method_time
    def rgb(self):
        return cvtColor(self.__img, COLOR_BGR2RGB)

    @Decorators.Loggers.log_class_method_time
    def gray(self):
        return cvtColor(self.__img, COLOR_BGR2GRAY)

    @staticmethod
    def _get_color(gray):
        return IMREAD_GRAYSCALE \
            if gray \
            else COLOR_BGR2RGB

    @staticmethod
    def save(image, name):
        try:
            imwrite(f"{DataPath.OUTPUT_PATH.value}/{name}", image)
        except Exception as e:
            print(e)

    def decode(self):
        try:
            return imdecode(
                np.fromstring(self.__file, np.uint8),
                Image._get_color(self.__gray)
            )
        except Exception as e:
            print(e)
            return None
