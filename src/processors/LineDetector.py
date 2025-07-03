import cv2
import numpy as np

from util.Processor import Processor


class LineDetector(Processor):
    def __init__(self, image, min_line_length=50, max_line_gap=10, line_thickness=3):
        super().__init__(image)
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.line_thickness = line_thickness

    def main(self, *args, **kwargs):
        gray = self._image.gray()
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)

        line_mask = np.zeros_like(gray, dtype=np.float32)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 1.0, thickness=self.line_thickness)

        self._image = line_mask