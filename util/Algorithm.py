from abc import abstractmethod


class Algorithm:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self._main(*args, **kwargs)

    @abstractmethod
    def _main(self, *args, **kwargs):
        pass

# Sobel + Disconnected Seam Carving (Implemented) - 90%
# Canny + Connected Seam Carving (Library | Github Code) - 50%

# Phase 2:
#
