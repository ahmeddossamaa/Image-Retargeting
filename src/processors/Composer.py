import cv2
import numpy as np

from config.decorators import Decorators
from config.plotter import Plotter
from src.processors.Midas import Midas
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from util.Processor import Processor


class Composer(Processor):
    @Decorators.Loggers.log_class_method_time
    def main(self, edge_weight=1.0, saliency_weight=1.5, depth_weight=0.8,
             show_debug=False, *args, **kwargs):
        self._depth = Midas(self._image)().image()
        self._saliency = SaliencyMap(self._image)().image()
        self._gradient = SobelFilter(self._image)().image()

        self._image = create_energy_map(self._depth, self._saliency, self._gradient,
                                        edge_weight=edge_weight,
                                        saliency_weight=saliency_weight,
                                        depth_weight=depth_weight,
                                        show_debug=show_debug)

    def get_depth(self):
        return self._depth

    def get_saliency(self):
        return self._saliency

    def get_gradient(self):
        return self._gradient


def normalize_to_range(img, target_min=0.0, target_max=1.0):
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val < 1e-8:
        return np.full_like(img, target_min)
    normalized = (img - min_val) / (max_val - min_val)
    return normalized * (target_max - target_min) + target_min


def calculate_edge_energy(gradient_img):
    normalized_gradient = normalize_to_range(gradient_img, 0.0, 1.0)
    return normalized_gradient


def calculate_saliency_energy(saliency_img):
    normalized_saliency = normalize_to_range(saliency_img, 0.0, 1.0)
    return normalized_saliency


def calculate_depth_energy(depth_img):
    normalized_depth = normalize_to_range(depth_img, 0.0, 1.0)
    # inverted_depth = 1.0 - normalized_depth
    return normalized_depth


def combine_energy_components(edge_energy, saliency_energy, depth_energy,
                              edge_weight, saliency_weight, depth_weight):
    weighted_edge = edge_energy * edge_weight
    weighted_saliency = saliency_energy * saliency_weight
    weighted_depth = depth_energy * depth_weight

    combined_energy = weighted_edge + weighted_saliency + weighted_depth
    return combined_energy


def apply_smooth_transitions(energy_map, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    smoothed = cv2.filter2D(energy_map, -1, kernel)
    return smoothed


def enhance_local_contrast(energy_map, alpha=0.3):
    blurred = cv2.GaussianBlur(energy_map.astype(np.float32), (5, 5), 1.0)
    enhanced = energy_map + alpha * (energy_map - blurred)
    return np.clip(enhanced, 0, np.max(energy_map))


def finalize_energy_map(energy_map):
    smoothed = apply_smooth_transitions(energy_map)
    enhanced = enhance_local_contrast(smoothed)
    final_energy = normalize_to_range(enhanced, 0.0, 255.0)
    return final_energy.astype(np.float32)


def create_energy_map(depth, saliency, gradient, edge_weight=1.0, saliency_weight=1.5,
                      depth_weight=0.8, show_debug=False):
    edge_energy = calculate_edge_energy(gradient)
    saliency_energy = calculate_saliency_energy(saliency)
    depth_energy = calculate_depth_energy(depth)

    combined_energy = combine_energy_components(edge_energy, saliency_energy, depth_energy,
                                                edge_weight, saliency_weight, depth_weight)

    final_energy = finalize_energy_map(combined_energy)

    if show_debug:
        print(f"Edge weight: {edge_weight}, Saliency weight: {saliency_weight}, Depth weight: {depth_weight}")
        print(f"Energy range: {np.min(final_energy):.2f} - {np.max(final_energy):.2f}")

        Plotter.images([
            normalize_to_range(depth, 0, 255),
            normalize_to_range(saliency, 0, 255),
            normalize_to_range(gradient, 0, 255),
            normalize_to_range(edge_energy, 0, 255),
            normalize_to_range(saliency_energy, 0, 255),
            normalize_to_range(depth_energy, 0, 255),
            normalize_to_range(combined_energy, 0, 255),
            final_energy
        ], 2, 4)

    return final_energy


def simple_energy_map(depth, saliency, gradient):
    return create_energy_map(depth, saliency, gradient,
                             edge_weight=1.0, saliency_weight=1.5, depth_weight=0.8,
                             show_debug=True)