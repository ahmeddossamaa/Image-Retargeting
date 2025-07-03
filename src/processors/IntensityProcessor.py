from config.helper import Helper
from util.Processor import Processor


class IntensityProcessor(Processor):
    def main(self, *args, **kwargs):
        image_array = self._image()

        if len(image_array.shape) == 3:
            intensity = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]
        else:
            intensity = image_array

        self._image = Helper.Math.normalize(intensity, 0, 1)