import io

import PIL.Image
import numpy as np


class Helper:
    class Image:
        """
        Used to convert rgb pixel to grayscale
        """
        @staticmethod
        def NTSC_formula(r, g, b) -> float:
            return 0.299 * r + 0.587 * g + 0.114 * b

        @staticmethod
        def get_stream(img, mimetype):
            image = PIL.Image.fromarray(img)

            stream = io.BytesIO()

            image.save(stream, format=mimetype.split('/')[1])

            return stream.getvalue()

    class Lists:
        @staticmethod
        def sort_with_indices(lst):
            sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
            sorted_indices = [index for index, _ in sorted_lst]
            return sorted_lst, sorted_indices

    class Math:
        @staticmethod
        def sigmoid(x):
            return 1 / 1 + np.exp(x)

        @staticmethod
        def scale(x, start=0.0, end=255.0):
            return (x - start) / (end - start)
