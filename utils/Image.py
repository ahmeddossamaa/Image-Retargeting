import numpy as np
from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_UNCHANGED, imwrite, imdecode, IMREAD_COLOR, COLOR_BGR2RGB, cvtColor, \
    COLOR_BGR2GRAY, COLOR_RGB2BGR, COLOR_BGR2LAB, split, COLOR_RGB2GRAY

from config.constants import DataPath
from config.decorators import Decorators
from config.helper import Helper


class Image:
    def __init__(self, file, gray=False, decode=False):
        self.__file = file
        self.__gray = gray

        self.__img = self.__read() if not decode else self.__decode()

    def __call__(self, *args, **kwargs):
        return self.__img.copy()

    @Decorators.Loggers.log_class_method_time
    def __read(self):
        img = imread(
            self.__file,
            Image._get_color(self.__gray)
        )

        if img is None:
            raise ValueError("Image not found")

        return img

    @Decorators.Loggers.log_class_method_time
    def __decode(self):
        try:
            return imdecode(
                np.fromstring(self.__file, np.uint8),
                Image._get_color(self.__gray)
            )
        except Exception as e:
            print(e)
            return None

    @Decorators.Loggers.log_class_method_time
    def copy(self):
        return self.__img.copy()

    @staticmethod
    @Decorators.Loggers.log_class_method_time
    def split(image):
        return split(image)

    @staticmethod
    @Decorators.Loggers.log_class_method_time
    def save(image, name):
        try:
            imwrite(f"{DataPath.OUTPUT_PATH.value}/{name}", cvtColor(image, COLOR_BGR2RGB))
        except Exception as e:
            print(e)

    @Decorators.Loggers.log_class_method_time
    def rgb(self):
        return cvtColor(self.__img, COLOR_BGR2RGB)

    @Decorators.Loggers.log_class_method_time
    def gray(self):
        print(self.__img)
        return cvtColor(self.__img, COLOR_BGR2GRAY)

    @Decorators.Loggers.log_class_method_time
    def lab(self):
        return cvtColor(self.__img, COLOR_BGR2LAB)

    @staticmethod
    def _get_color(gray):
        return IMREAD_GRAYSCALE \
            if gray \
            else COLOR_BGR2RGB
