from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_UNCHANGED, imwrite

from config.constants import DataPath
from config.decorators import Decorators
from config.helper import Helper


class Image:
    def __init__(self, path, gray=True):
        self.__path = path
        self.__gray = gray

        self.__img = self.__read()

    def __call__(self, *args, **kwargs):
        return self.__img.copy()

    @Decorators.Loggers.log_class_method_time
    def __read(self):
        return imread(
            self.__path,
            IMREAD_GRAYSCALE
            if self.__gray
            else IMREAD_UNCHANGED
        )

    @staticmethod
    def save(image, name):
        try:
            imwrite(f"{DataPath.OUTPUT_PATH.value}/{name}", image)
        except Exception as e:
            print(e)
