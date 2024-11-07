import cv2
import numpy as np

from config.decorators import Decorators
from config.plotter import Plotter
from src.processors.Midas import Midas
from src.processors.SobelFilter import SobelFilter
from src.processors.UNetSaliency import UNetSaliency
from util.Processor import Processor


class Fucker(Processor):
    @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        depth = Midas(self._image)().image()
        saliency = UNetSaliency(self._image)().image()
        gradient = SobelFilter(self._image)().image()

        self._image = multi_scale_fusion(depth, saliency, gradient)

        # Plotter.images([depth, saliency, gradient, self._image], 1, 4)


def multi_scale_fusion(depth, saliency, gradient, rcnn_mask=None, levels=5, epsilon=1e-6):
    def build_gaussian_pyramid(img, levels):
        pyramid = [img]
        for i in range(levels - 1):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid

    def normalize_map(map):
        min_val = np.min(map)
        max_val = np.max(map)
        if max_val - min_val > epsilon:
            return (map - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(map)

    # Normalize inputs
    inputs = [depth, saliency, gradient]
    normalized_inputs = [normalize_map(img) for img in inputs]

    pyramids = [build_gaussian_pyramid(img, levels) for img in normalized_inputs]
    fused_pyramid = []

    for level in range(levels):
        level_maps = [pyr[level] for pyr in pyramids]

        # Check if any map is all zeros (or very close to zero)
        non_zero_maps = [map for map in level_maps if np.mean(map) > epsilon]

        if non_zero_maps:
            # Use only non-zero maps for fusion
            weights = [1] * len(non_zero_maps)
            weighted_sum = sum(w * map for w, map in zip(weights, non_zero_maps))
            fused_level = weighted_sum / sum(weights)
        else:
            # If all maps are zero, use a uniform map
            fused_level = np.ones_like(level_maps[0]) * 0.5

        fused_pyramid.append(fused_level)

    fused = fused_pyramid[-1]
    for i in range(levels - 2, -1, -1):
        fused = cv2.pyrUp(fused)
        fused = cv2.resize(fused, (fused_pyramid[i].shape[1], fused_pyramid[i].shape[0]))
        fused = cv2.add(fused, fused_pyramid[i])

    # Ensure the final fused image is the same size as the input
    fused = cv2.resize(fused, (depth.shape[1], depth.shape[0]))

    fused = normalize_map(fused)
    # fused =
    return fused
