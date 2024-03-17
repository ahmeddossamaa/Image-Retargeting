from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_UNCHANGED, imwrite

from config.constants import DataPath
from config.decorators import Decorators
from config.helper import Helper


class Image:
    def __init__(self, path, gray=True):
        self.__path = path
        self.__gray = gray

        self.__img = self.__read()

    def load(self):
        # Load image data from file
        # Implementation depends on the library you're using to handle images
        pass

    def copy(self):
        # Create a copy of the image object
        copy_image = Image(self.__path, self.__gray)
        copy_image.__img = self.__img.copy()  # Assuming __img is the attribute containing image data
        return copy_image
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
