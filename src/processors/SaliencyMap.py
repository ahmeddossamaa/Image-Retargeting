import cv2
import numpy as np
from config.helper import Helper
from util.Processor import Processor


class SaliencyMap(Processor):
    def __init__(self, image, method='fine_grained'):
        super().__init__(image)
        self.method = method

    def main(self, *args, **kwargs):
        image_array = self._image()

        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        if self.method == 'spectral':
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        elif self.method == 'fine_grained':
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        success, saliency_map = saliency.computeSaliency(image_array)

        if success:
            saliency_normalized = Helper.Math.normalize(saliency_map, 0, 1)
        else:
            # Fallback to intensity if saliency fails
            if len(image_array.shape) == 3:
                intensity = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]
            else:
                intensity = image_array
            saliency_normalized = Helper.Math.normalize(intensity, 0, 1)

        self._image = saliency_normalized
