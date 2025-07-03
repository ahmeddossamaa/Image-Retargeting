import cv2
import numpy as np
from scipy import ndimage
from config.decorators import Decorators
from src.processors.Midas import Midas
from src.processors.SobelFilter import SobelFilter
from src.processors.UNetSaliency import UNetSaliency
from util.Processor import Processor


class OptimalFusionProcessor(Processor):
    def __init__(self, image, fusion_strategy='adaptive', base_weights=None):
        """
        Optimal fusion processor with mathematically-driven weighting

        Args:
            image: Input image
            fusion_strategy: 'adaptive', 'proximity_priority', 'saliency_priority', 'detail_preserving'
            base_weights: Base weights [depth, saliency, gradient] if using custom strategy
        """
        super().__init__(image)
        self.fusion_strategy = fusion_strategy
        self.base_weights = base_weights or [0.5, 0.35, 0.15]  # Default proximity-focused
        self.epsilon = 1e-8

    @Decorators.Loggers.log_class_method_time
    def main(self, *args, **kwargs):
        # Generate feature maps
        depth = Midas(self._image)().image()
        saliency = UNetSaliency(self._image)().image()
        gradient = SobelFilter(self._image)().image()

        # Apply optimal fusion based on strategy
        self._image = self.optimal_fusion(depth, saliency, gradient)

    def analyze_map_characteristics(self, depth, saliency, gradient):
        """
        Analyze each map's characteristics for informed weighting

        Mathematical analysis:
        1. Depth: Focus on proximity (inverse depth values indicate closer objects)
        2. Saliency: Focus on visual attention areas
        3. Gradient: Focus on edge preservation and fine details
        """
        analysis = {}

        # Normalize inputs
        depth_norm = self.robust_normalize(depth)
        saliency_norm = self.robust_normalize(saliency)
        gradient_norm = self.robust_normalize(gradient)

        # DEPTH ANALYSIS
        # Near objects have low depth values (high inverse depth)
        inverse_depth = 1.0 - depth_norm  # Invert so near objects have high values
        proximity_strength = np.mean(inverse_depth > 0.7)  # Percentage of near objects
        depth_variance = np.var(inverse_depth)

        analysis['depth'] = {
            'proximity_strength': proximity_strength,  # How much foreground content
            'variance': depth_variance,  # How varied the depth is
            'quality': min(proximity_strength + depth_variance, 1.0)
        }

        # SALIENCY ANALYSIS
        # High saliency indicates important objects
        saliency_peak = np.percentile(saliency_norm, 90)  # Top 10% saliency values
        saliency_coverage = np.mean(saliency_norm > 0.5)  # Coverage of salient areas
        saliency_focus = np.std(saliency_norm)  # How focused the saliency is

        analysis['saliency'] = {
            'peak_strength': saliency_peak,
            'coverage': saliency_coverage,
            'focus': saliency_focus,
            'quality': min(saliency_peak * saliency_focus, 1.0)
        }

        # GRADIENT ANALYSIS
        # High gradient indicates edges and fine details
        edge_density = np.mean(gradient_norm > 0.3)  # Percentage of edge pixels
        detail_richness = np.std(gradient_norm)  # Variance in edge strengths
        fine_detail_score = self.calculate_fine_detail_score(gradient_norm)

        analysis['gradient'] = {
            'edge_density': edge_density,
            'detail_richness': detail_richness,
            'fine_detail_score': fine_detail_score,
            'quality': min(edge_density + detail_richness, 1.0)
        }

        return analysis, inverse_depth, saliency_norm, gradient_norm

    def calculate_fine_detail_score(self, gradient_map):
        """Calculate how much fine detail is present"""
        # Use Laplacian to detect fine details
        laplacian = cv2.Laplacian(gradient_map, cv2.CV_64F)
        fine_details = np.sum(np.abs(laplacian) > 0.1) / gradient_map.size
        return min(fine_details, 1.0)

    def compute_optimal_weights(self, analysis):
        """
        Compute mathematically optimal weights based on your goals:
        1. Priority for near objects (depth)
        2. Highlighting salient objects (saliency)
        3. Don't ignore small details (gradient)
        """

        if self.fusion_strategy == 'proximity_priority':
            # Strategy 1: Proximity-focused (your main goal)
            # Weight = base_weight * (1 + quality_boost + proximity_boost)

            depth_boost = analysis['depth']['proximity_strength'] * 0.3  # Boost if many near objects
            saliency_boost = analysis['saliency']['peak_strength'] * 0.2  # Boost if strong saliency
            detail_boost = analysis['gradient']['fine_detail_score'] * 0.15  # Boost if rich details

            weights = [
                self.base_weights[0] * (1 + depth_boost),  # 0.5 base, up to 0.65
                self.base_weights[1] * (1 + saliency_boost),  # 0.35 base, up to 0.42
                self.base_weights[2] * (1 + detail_boost)  # 0.15 base, up to 0.17
            ]

        elif self.fusion_strategy == 'adaptive':
            # Strategy 2: Fully adaptive based on content
            depth_quality = analysis['depth']['quality']
            saliency_quality = analysis['saliency']['quality']
            gradient_quality = analysis['gradient']['quality']

            # Use softmax-like distribution with temperature control
            temperature = 2.0  # Controls how "sharp" the distribution is
            raw_weights = [
                depth_quality * 2.0,  # 2x multiplier for depth priority
                saliency_quality * 1.5,  # 1.5x multiplier for saliency
                gradient_quality * 1.0  # 1x multiplier for gradients
            ]

            # Apply softmax with temperature
            exp_weights = [np.exp(w / temperature) for w in raw_weights]
            sum_exp = sum(exp_weights)
            weights = [w / sum_exp for w in exp_weights]

        elif self.fusion_strategy == 'detail_preserving':
            # Strategy 3: Ensure small details aren't lost
            min_gradient_weight = 0.25  # Minimum weight for gradient preservation

            total_for_others = 1.0 - min_gradient_weight
            depth_ratio = analysis['depth']['quality'] / (
                        analysis['depth']['quality'] + analysis['saliency']['quality'] + self.epsilon)
            saliency_ratio = 1.0 - depth_ratio

            weights = [
                total_for_others * depth_ratio * 1.4,  # Favor depth
                total_for_others * saliency_ratio,  # Remaining for saliency
                min_gradient_weight  # Guaranteed minimum for details
            ]

        else:  # custom weights
            weights = self.base_weights

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Apply safety bounds
        weights = [max(0.05, min(0.7, w)) for w in weights]  # Clamp between 5% and 70%

        # Renormalize after clamping
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return weights

    def apply_spatial_modulation(self, maps, weights, analysis):
        """
        Apply spatially-varying weights based on local characteristics
        This ensures near objects and salient regions get proper attention
        """
        depth_map, saliency_map, gradient_map = maps
        depth_weight, saliency_weight, gradient_weight = weights

        h, w = depth_map.shape

        # Create spatial modulation masks

        # 1. Proximity mask (boost near objects)
        proximity_mask = 1.0 - depth_map  # Inverted depth (near = high)
        proximity_mask = cv2.GaussianBlur(proximity_mask, (15, 15), 5)  # Smooth transitions

        # 2. Saliency attention mask (boost salient regions)
        saliency_attention = cv2.GaussianBlur(saliency_map, (21, 21), 7)
        saliency_threshold = np.percentile(saliency_attention, 70)
        saliency_mask = np.clip(saliency_attention / (saliency_threshold + self.epsilon), 0, 2)

        # 3. Detail preservation mask (boost high-frequency areas)
        detail_mask = cv2.GaussianBlur(gradient_map, (5, 5), 1)
        detail_boost = np.clip(detail_mask * 1.5, 0.5, 2.0)

        # Apply spatial modulation
        spatial_depth_weight = depth_weight * (1 + 0.3 * proximity_mask)  # Up to 30% boost for near objects
        spatial_saliency_weight = saliency_weight * (1 + 0.4 * saliency_mask)  # Up to 40% boost for salient areas
        spatial_gradient_weight = gradient_weight * detail_boost  # Variable boost for details

        # Normalize spatially-varying weights
        total_spatial = spatial_depth_weight + spatial_saliency_weight + spatial_gradient_weight
        spatial_depth_weight /= (total_spatial + self.epsilon)
        spatial_saliency_weight /= (total_spatial + self.epsilon)
        spatial_gradient_weight /= (total_spatial + self.epsilon)

        return [spatial_depth_weight, spatial_saliency_weight, spatial_gradient_weight]

    def optimal_fusion(self, depth, saliency, gradient):
        """Main fusion function with optimal weighting"""

        # Analyze input characteristics
        analysis, depth_norm, saliency_norm, gradient_norm = self.analyze_map_characteristics(depth, saliency, gradient)

        # Compute optimal weights
        optimal_weights = self.compute_optimal_weights(analysis)

        print(f"Optimal weights - Depth: {optimal_weights[0]:.3f}, "
              f"Saliency: {optimal_weights[1]:.3f}, Gradient: {optimal_weights[2]:.3f}")

        # Apply spatial modulation for locally-adaptive fusion
        spatial_weights = self.apply_spatial_modulation(
            [depth_norm, saliency_norm, gradient_norm],
            optimal_weights,
            analysis
        )

        # Perform spatially-adaptive fusion
        fused_result = (spatial_weights[0] * depth_norm +
                        spatial_weights[1] * saliency_norm +
                        spatial_weights[2] * gradient_norm)

        # Post-processing to enhance the fusion result
        fused_result = self.post_process_fusion(fused_result, depth_norm, saliency_norm)

        return self.robust_normalize(fused_result)

    def post_process_fusion(self, fused, depth, saliency):
        """Post-process the fusion result to enhance key areas"""

        # 1. Enhance near-object boundaries
        near_objects = (1.0 - depth) > 0.6  # Binary mask of near objects
        near_boundaries = cv2.Canny((near_objects * 255).astype(np.uint8), 50, 150) / 255.0

        # 2. Enhance salient region cores
        saliency_cores = saliency > np.percentile(saliency, 80)
        saliency_cores = ndimage.binary_erosion(saliency_cores, iterations=2)

        # Apply enhancements
        enhancement = np.zeros_like(fused)
        enhancement += near_boundaries * 0.1  # Boost near object edges
        enhancement += saliency_cores * 0.15  # Boost salient cores

        enhanced = fused + enhancement
        return np.clip(enhanced, 0, 1)

    def robust_normalize(self, map_data):
        """Robust normalization with outlier handling"""
        if map_data is None or map_data.size == 0:
            return np.zeros_like(map_data) if map_data is not None else None

        # Clip extreme outliers
        p1, p99 = np.percentile(map_data, [1, 99])
        clipped = np.clip(map_data, p1, p99)

        min_val = np.min(clipped)
        max_val = np.max(clipped)

        if max_val - min_val > self.epsilon:
            return (clipped - min_val) / (max_val - min_val)
        else:
            return np.full_like(clipped, 0.5)


# Comparison function to test different strategies
def test_fusion_strategies(image_path):
    """Test different fusion strategies and return results"""
    from util.Image import Image

    img = Image(image_path)

    strategies = {
        'proximity_priority': OptimalFusionProcessor(img, 'proximity_priority'),
        'adaptive': OptimalFusionProcessor(img, 'adaptive'),
        'detail_preserving': OptimalFusionProcessor(img, 'detail_preserving'),
        'custom_balanced': OptimalFusionProcessor(img, 'custom', [0.4, 0.4, 0.2])
    }

    results = {}
    for name, processor in strategies.items():
        results[name] = processor().image()

    return results


# Mathematical weight optimization function
def find_optimal_weights_mathematically(depth_chars, saliency_chars, gradient_chars, priorities):
    """
    Mathematical optimization of weights based on:
    - priorities: dict with 'proximity', 'saliency', 'detail_preservation' importance (0-1)
    - characteristics: quality measures of each modality

    Uses constrained optimization to find weights that maximize:
    objective = Σ(priority_i * quality_i * weight_i) - λ * penalty_for_extreme_weights
    """
    from scipy.optimize import minimize

    def objective(weights):
        w_depth, w_saliency, w_gradient = weights

        # Primary objective: weighted quality scores
        score = (priorities['proximity'] * depth_chars['quality'] * w_depth +
                 priorities['saliency'] * saliency_chars['quality'] * w_saliency +
                 priorities['detail_preservation'] * gradient_chars['quality'] * w_gradient)

        # Penalty for extreme weights (encourages balanced solutions)
        penalty = 0.1 * (w_depth ** 2 + w_saliency ** 2 + w_gradient ** 2)

        return -(score - penalty)  # Negative because minimize() finds minimum

    # Constraints: weights sum to 1, all positive, reasonable bounds
    constraints = [
        {'type': 'eq', 'fun': lambda w: w[0] + w[1] + w[2] - 1},  # Sum to 1
    ]
    bounds = [(0.05, 0.7), (0.05, 0.7), (0.05, 0.7)]  # Reasonable bounds

    # Initial guess based on priorities
    initial_guess = [
        priorities['proximity'] * 0.6,
        priorities['saliency'] * 0.6,
        priorities['detail_preservation'] * 0.6
    ]
    # Normalize initial guess
    total = sum(initial_guess)
    initial_guess = [w / total for w in initial_guess]

    result = minimize(objective, initial_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    return result.x if result.success else initial_guess