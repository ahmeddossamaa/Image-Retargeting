import numpy as np
from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.MiddleSCI import MiddleSCI
from util.Image import Image


class DynamicSC(MiddleSCI):
    def __init__(self, img, ratio, converter=None, feature_map=None, prev_matrix: bool = False,
                 strategy='adaptive_change', **kwargs):
        super(DynamicSC, self).__init__(img, ratio, converter=converter,
                                                     feature_map=feature_map, prev_matrix=prev_matrix)

        self.strategy = strategy
        self.recompute_history = []
        self.last_energy_recompute = 0
        self.cumulative_change = 0.0
        self.energy_quality_threshold = kwargs.get('energy_threshold', 0.15)
        self.min_interval = kwargs.get('min_interval', 3)
        self.max_interval = kwargs.get('max_interval', 15)
        self.change_threshold = kwargs.get('change_threshold', 0.12)

    @Decorators.Loggers.log_class_method_time
    def execute(self):
        result = self._image.copy()
        original_energy = None

        for k in range(self._num_seams):
            should_recompute = self._should_recompute_energy(k, result)

            if should_recompute or k == 0:
                if k > 0:
                    print(f"Recomputing energy at seam {k} (strategy: {self.strategy})")

                self._origin = Image("", data=result)
                self._height, self._width, self._z = result.shape

                energy = self.convert(self._origin)
                if original_energy is None:
                    original_energy = energy.copy()

                self.accumulate(energy)

                # Reset tracking variables
                self.last_energy_recompute = k
                self.cumulative_change = 0.0
                self.recompute_history.append(k)

                middle = self._matrix[self._mid - 1] + self._matrix[self._mid]

            # Find and remove seam
            i = np.argmin(middle)
            middle = np.delete(middle, i, axis=0)

            # Track change for dynamic strategies
            if k > 0:
                change = self._measure_change(result, i)
                self.cumulative_change += change

            result = self.remove(energy, i)

        return np.array(result)

    def _should_recompute_energy(self, current_seam, current_image):
        """Determine if energy should be recomputed based on the selected strategy"""

        if current_seam == 0:
            return True

        seams_since_last = current_seam - self.last_energy_recompute

        if self.strategy == 'adaptive_change':
            return self._adaptive_change_strategy(current_seam, seams_since_last)
        elif self.strategy == 'energy_variance':
            return self._energy_variance_strategy(current_seam, seams_since_last)
        elif self.strategy == 'seam_quality':
            return self._seam_quality_strategy(current_seam, seams_since_last)
        elif self.strategy == 'hybrid':
            return self._hybrid_strategy(current_seam, seams_since_last, current_image)
        else:
            # Fallback to improved fixed rate
            return self._improved_fixed_rate(current_seam, seams_since_last)

    def _adaptive_change_strategy(self, current_seam, seams_since_last):
        """Recompute based on cumulative image changes"""

        # Ensure minimum interval
        if seams_since_last < self.min_interval:
            return False

        # Force recomputation if max interval reached
        if seams_since_last >= self.max_interval:
            return True

        # Check if cumulative change exceeds threshold
        return self.cumulative_change > self.change_threshold

    def _energy_variance_strategy(self, current_seam, seams_since_last):
        """Recompute based on energy distribution degradation"""

        if seams_since_last < self.min_interval:
            return False

        if seams_since_last >= self.max_interval:
            return True

        # Check energy variance in the middle selection area
        if hasattr(self, '_matrix') and self._matrix is not None:
            middle_energy = self._matrix[self._mid - 1] + self._matrix[self._mid]
            energy_std = np.std(middle_energy)
            energy_mean = np.mean(middle_energy)

            # If variance is too low, energy map might be degraded
            coefficient_of_variation = energy_std / (energy_mean + 1e-8)
            return coefficient_of_variation < self.energy_quality_threshold

        return False

    def _seam_quality_strategy(self, current_seam, seams_since_last):
        """Recompute based on seam selection quality"""

        if seams_since_last < self.min_interval:
            return False

        if seams_since_last >= self.max_interval:
            return True

        # Analyze recent seam positions for clustering (bad sign)
        if len(self.recompute_history) >= 2:
            recent_interval = current_seam - self.recompute_history[-1]
            prev_interval = self.recompute_history[-1] - self.recompute_history[-2]

            # If intervals are decreasing rapidly, quality might be degrading
            if recent_interval < prev_interval * 0.7:
                return True

        return False

    def _hybrid_strategy(self, current_seam, seams_since_last, current_image):
        """Combine multiple strategies for robust decision making"""

        if seams_since_last < self.min_interval:
            return False

        if seams_since_last >= self.max_interval:
            return True

        # Score-based approach
        recompute_score = 0.0

        # Change-based component
        if self.cumulative_change > self.change_threshold * 0.8:
            recompute_score += 0.4

        # Energy variance component
        if hasattr(self, '_matrix') and self._matrix is not None:
            middle_energy = self._matrix[self._mid - 1] + self._matrix[self._mid]
            energy_std = np.std(middle_energy)
            energy_mean = np.mean(middle_energy)
            coefficient_of_variation = energy_std / (energy_mean + 1e-8)

            if coefficient_of_variation < self.energy_quality_threshold:
                recompute_score += 0.3

        # Interval-based pressure (increases over time)
        interval_pressure = min(1.0, seams_since_last / self.max_interval)
        recompute_score += interval_pressure * 0.3

        return recompute_score > 0.6

    def _improved_fixed_rate(self, current_seam, seams_since_last):
        """Improved fixed rate with dynamic adjustment"""

        # Adaptive rate based on image complexity and progress
        progress_ratio = current_seam / self._num_seams

        # Start with more frequent updates, then space them out
        if progress_ratio < 0.3:
            target_interval = 3
        elif progress_ratio < 0.7:
            target_interval = 5
        else:
            target_interval = 8

        return seams_since_last >= target_interval

    def _measure_change(self, current_image, removed_seam_idx):
        """Measure the impact of removing a seam"""

        # Simple change metric based on seam position variability
        height = current_image.shape[0]

        # If seams are being removed from similar positions, change is high
        if hasattr(self, '_recent_seam_positions'):
            self._recent_seam_positions.append(removed_seam_idx)
            if len(self._recent_seam_positions) > 5:
                self._recent_seam_positions.pop(0)

            # Calculate position variance
            position_variance = np.var(self._recent_seam_positions)
            max_possible_variance = (self._width ** 2) / 4

            # Normalize and invert (low variance = high change impact)
            normalized_variance = position_variance / max_possible_variance
            change_metric = 1.0 - min(1.0, normalized_variance)

            return change_metric * 0.1  # Scale appropriately
        else:
            self._recent_seam_positions = [removed_seam_idx]
            return 0.05

    def remove(self, energy, i):
        """Modified remove method that preserves the matrix state"""
        self._new_image = [[] for _ in range(self._height)]
        self._new_matrix = [[] for _ in range(self._height)]

        def bottomup(i, s, e, step):
            for j in range(s, e, step):
                v, i = Helper.Math.get_min(self._matrix, j, i, 0, len(self._matrix[j]) - 1)
                self._new_image[j] = np.delete(self._image[j], i, axis=0)
                self._new_matrix[j] = np.delete(self._matrix[j], i, axis=0)

        bottomup(i, self._mid - 1, -1, -1)
        bottomup(i, self._mid, self._height, 1)

        self._image = self._new_image
        self._matrix = self._new_matrix

        return np.array(self._new_image)

    def get_recompute_stats(self):
        """Get statistics about recomputation frequency"""
        if len(self.recompute_history) <= 1:
            return {"intervals": [], "avg_interval": 0, "total_recomputes": len(self.recompute_history)}

        intervals = [self.recompute_history[i] - self.recompute_history[i - 1]
                     for i in range(1, len(self.recompute_history))]

        return {
            "intervals": intervals,
            "avg_interval": np.mean(intervals),
            "total_recomputes": len(self.recompute_history),
            "recompute_positions": self.recompute_history
        }