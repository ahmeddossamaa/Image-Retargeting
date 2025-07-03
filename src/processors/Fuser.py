import cv2
import numpy as np

from config.decorators import Decorators
from config.helper import Helper
from src.processors.DepthAnythingV2 import DepthAnythingV2
from src.processors.SaliencyMap import SaliencyMap
from src.processors.SobelFilter import SobelFilter
from src.processors.UNetSaliency import UNetSaliency
from util.Processor import Processor


class Fuser(Processor):
    @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        self._depth = DepthAnythingV2(self._image)().image()
        self._saliency = UNetSaliency(self._image)().image()
        self._gradient = SobelFilter(self._image)().image()

        # Quality assessment
        depth_quality = self._assess_depth_quality(self._depth)
        saliency_quality = self._assess_saliency_quality(self._saliency)
        gradient_quality = self._assess_gradient_quality(self._gradient)

        # Content analysis
        content_type = self._analyze_content_type()

        # Adaptive weighting
        weights = self._calculate_adaptive_weights(depth_quality, saliency_quality, gradient_quality, content_type)

        # Multi-scale fusion
        self._image = self._multi_scale_fusion(self._depth, self._saliency, self._gradient, weights)

    def _assess_depth_quality(self, depth_map):
        variance = np.var(depth_map)
        edge_density = np.mean(cv2.Canny((depth_map * 255).astype(np.uint8), 50, 150))
        extreme_ratio = np.mean((depth_map < 0.05) | (depth_map > 0.95))

        quality = variance * 0.4 + edge_density * 0.4 - extreme_ratio * 0.2
        return max(0, min(1, quality))

    def _assess_saliency_quality(self, saliency_map):
        contrast = np.std(saliency_map)
        peak_ratio = np.mean(saliency_map > 0.7)
        edge_coherence = np.mean(cv2.Canny((saliency_map * 255).astype(np.uint8), 100, 200))

        quality = contrast * 0.5 + peak_ratio * 0.3 + edge_coherence * 0.2
        return max(0, min(1, quality))

    def _assess_gradient_quality(self, gradient_map):
        edge_density = np.mean(gradient_map > 0.1)
        distribution = 1.0 - np.abs(0.5 - np.mean(gradient_map))
        connectivity = self._measure_edge_connectivity(gradient_map)

        quality = edge_density * 0.4 + distribution * 0.3 + connectivity * 0.3
        return max(0, min(1, quality))

    def _measure_edge_connectivity(self, gradient_map):
        binary_edges = (gradient_map > 0.2).astype(np.uint8)
        contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_length = sum(cv2.arcLength(c, False) for c in contours)
        return min(1.0, total_length / (gradient_map.shape[0] * gradient_map.shape[1]) * 100)

    def _analyze_content_type(self):
        image_array = self._image()

        # Texture analysis
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        texture_variance = np.var(cv2.Laplacian(gray, cv2.CV_64F))

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges)

        # Brightness distribution
        brightness_std = np.std(gray)

        if texture_variance > 1000 and edge_density > 30:
            return 'high_texture'
        elif brightness_std < 40 and edge_density < 20:
            return 'uniform'
        elif edge_density > 50:
            return 'architectural'
        else:
            return 'natural'

    def _calculate_adaptive_weights(self, depth_q, saliency_q, gradient_q, content_type):
        base_weights = {
            'high_texture': [0.2, 0.3, 0.5],
            'uniform': [0.4, 0.5, 0.1],
            'architectural': [0.5, 0.2, 0.3],
            'natural': [0.3, 0.5, 0.2]
        }

        weights = base_weights.get(content_type, [0.33, 0.33, 0.34])
        quality_factors = [depth_q, saliency_q, gradient_q]

        # Adjust weights based on quality
        adjusted_weights = [w * (0.5 + 0.5 * q) for w, q in zip(weights, quality_factors)]

        # Normalize
        total = sum(adjusted_weights)
        return [w / total for w in adjusted_weights]

    def _multi_scale_fusion(self, depth, saliency, gradient, weights, levels=4):
        def build_pyramid(img, levels):
            pyramid = [img]
            for i in range(levels - 1):
                img = cv2.pyrDown(img)
                pyramid.append(img)
            return pyramid

        # Build pyramids
        depth_pyramid = build_pyramid(depth, levels)
        saliency_pyramid = build_pyramid(saliency, levels)
        gradient_pyramid = build_pyramid(gradient, levels)

        fused_pyramid = []

        for level in range(levels):
            d_level = depth_pyramid[level]
            s_level = saliency_pyramid[level]
            g_level = gradient_pyramid[level]

            # Weighted combination at each level
            fused_level = (weights[0] * d_level +
                           weights[1] * s_level +
                           weights[2] * g_level)

            fused_pyramid.append(fused_level)

        # Reconstruct from pyramid
        result = fused_pyramid[-1]
        for i in range(levels - 2, -1, -1):
            result = cv2.pyrUp(result)
            result = cv2.resize(result, (fused_pyramid[i].shape[1], fused_pyramid[i].shape[0]))
            result = cv2.addWeighted(result, 0.6, fused_pyramid[i], 0.4, 0)

        # Ensure correct size
        result = cv2.resize(result, (depth.shape[1], depth.shape[0]))

        # Final normalization and enhancement
        result = Helper.Math.normalize(result, 0, 1)
        return self._post_process_energy(result)

    def _post_process_energy(self, energy_map):
        # Enhance important boundaries
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(energy_map, -1, kernel)

        # Blend original with sharpened
        enhanced = cv2.addWeighted(energy_map, 0.7, sharpened, 0.3, 0)

        # Ensure [0,1] range
        return np.clip(enhanced, 0, 1)

    def get_depth(self):
        return self._depth

    def get_saliency(self):
        return self._saliency

    def get_gradient(self):
        return self._gradient