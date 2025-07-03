import cv2
import numpy as np

from config.helper import Helper
from util.Processor import Processor


class OrientationDetector(Processor):
    def __init__(self, image, vertical_threshold=0.7):
        super().__init__(image)
        self.vertical_threshold = vertical_threshold

    def main(self, *args, **kwargs):
        gray = self._image.gray()

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        vertical_edges = np.abs(np.cos(angle)) > self.vertical_threshold
        vertical_penalty = magnitude * vertical_edges

        self._image = Helper.Math.normalize(vertical_penalty, 0, 1)