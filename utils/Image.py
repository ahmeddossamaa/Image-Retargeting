from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_UNCHANGED

from config.decorators import Decorators
from config.helper import Helper


class Image:
    def __init__(self, path, gray=True):
        self.__path = path
        self.__gray = gray

        self.__img = self.__read()

    def __call__(self, *args, **kwargs):
        return self.__img.copy()

    @Decorators.log_class_method_time
    def __read(self):
        return imread(
            self.__path,
            IMREAD_GRAYSCALE
            if self.__gray
            else IMREAD_UNCHANGED
        )

    """
    Execution time is above 1 sec, wouldn't recommend it.
    """
    @Decorators.log_class_method_time
    def to_gray_scale(self):
        if self.__gray:
            return self.__img

        x, y, z = self.__img.shape

        return [
            [
                Helper.Image.NTSC_formula(
                    *self.__img[i, j]
                ) for j in range(y)
            ] for i in range(x)
        ]
