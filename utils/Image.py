from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_UNCHANGED, resize


class Image:
    __img = None

    def __init__(self, path, gray=True):
        self.__path = path
        self.__gray = gray

        self.__read()

    def __call__(self, *args, **kwargs):
        return self.__img

    def __read(self):
        self.__img = imread(
            self.__path,
            IMREAD_GRAYSCALE
            if self.__gray
            else IMREAD_UNCHANGED
        )
