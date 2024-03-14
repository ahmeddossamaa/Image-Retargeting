import io

import PIL.Image


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

            image_stream = io.BytesIO()

            image.save(image_stream, format=mimetype.split('/')[1])

            return image_stream.getvalue()

    class Lists:
        @staticmethod
        def sort_with_indices(lst):
            sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
            sorted_indices = [index for index, _ in sorted_lst]
            return sorted_lst, sorted_indices
