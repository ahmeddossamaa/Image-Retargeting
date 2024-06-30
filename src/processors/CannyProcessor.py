from util.Processor import Processor
from cv2 import Canny


class CannyProcessor(Processor):
    def main(self, *args, **kwargs):
        edges = Canny(self._origin.gray(), 150, 250)
        self._image = edges

