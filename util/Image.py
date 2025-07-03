import numpy as np
import cv2
from cv2 import imread, IMREAD_GRAYSCALE, imwrite, imdecode, COLOR_BGR2RGB, cvtColor, \
    COLOR_BGR2GRAY, COLOR_BGR2LAB, split


class Image:
    def __init__(self, file, gray=False, decode=False, data=None, max_width=1200):
        self.__file = file
        self.__gray = gray
        self.__data = data
        self.__max_width = max_width
        self.__original_size = None
        self.__scale_factor = 1.0

        self.__img = self.__read() if not decode else self.__decode()
        self.__img = self.__auto_resize()

    def __call__(self, *args, **kwargs):
        return self.__img.copy()

    def __read(self):
        img = imread(
            self.__file,
            Image._get_color(self.__gray)
        ) if self.__data is None else self.__data

        if img is None:
            raise ValueError(f"Image {self.__file} not found")

        return img

    def __decode(self):
        try:
            return imdecode(
                np.fromstring(self.__file, np.uint8),
                Image._get_color(self.__gray)
            )
        except Exception as e:
            print(e)
            return None

    def __auto_resize(self):
        if self.__img is None:
            return None

        self.__original_size = (self.__img.shape[1], self.__img.shape[0])  # (width, height)

        if self.__original_size[0] > self.__max_width:
            self.__scale_factor = self.__max_width / self.__original_size[0]
            new_width = self.__max_width
            new_height = int(self.__original_size[1] * self.__scale_factor)

            print(f"Resizing image from {self.__original_size} to ({new_width}, {new_height})")
            return cv2.resize(self.__img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return self.__img

    def copy(self):
        return self.__img.copy()

    def get_original_size(self):
        return self.__original_size

    def get_scale_factor(self):
        return self.__scale_factor

    def scale_back(self, processed_image, target_size=None):
        """Scale processed image back to original or target size"""
        if target_size is None:
            target_size = self.__original_size

        return cv2.resize(processed_image, target_size, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def split(image):
        return split(image)

    @staticmethod
    def save(image, name):
        try:
            imwrite(name, cvtColor(image, COLOR_BGR2RGB))
        except Exception as e:
            print(e)

    def rgb(self):
        return cvtColor(self.__img, COLOR_BGR2RGB)

    def gray(self):
        return cvtColor(self.__img, COLOR_BGR2GRAY)

    def lab(self):
        return cvtColor(self.__img, COLOR_BGR2LAB)

    @staticmethod
    def _get_color(gray):
        return IMREAD_GRAYSCALE \
            if gray \
            else COLOR_BGR2RGB