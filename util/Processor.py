from abc import abstractmethod

import numpy as np
import torch

from util.Image import Image


class Processor:
    def __init__(self, image: Image):
        self._image = image
        self._origin = image.copy()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self, *args, **kwargs):
        self.main(*args, **kwargs)

        return self

    @abstractmethod
    def main(self, *args, **kwargs):
        pass

    def origin(self):
        return self._origin

    def image(self):
        return self._image