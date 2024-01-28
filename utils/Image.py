from cv2 import imread


class Image:
    __path = ""
    __img = None

    def __init__(self, path):
        self.__path = path

        self.__read()

    def __call__(self, *args, **kwargs):
        return self.__img

    def __read(self):
        self.__img = imread(self.__path)
