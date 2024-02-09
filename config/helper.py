class Helper:
    class Image:
        """
        Used to convert rgb pixel to grayscale
        """
        @staticmethod
        def NTSC_formula(r, g, b) -> float:
            return 0.299 * r + 0.587 * g + 0.114 * b
