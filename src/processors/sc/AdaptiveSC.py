import math
import numpy as np
from typing import Tuple, List

from config.decorators import Decorators
from config.helper import Helper
from config.plotter import Plotter
from src.processors.sc.SeamCarving import SeamCarving
from src.processors.Fuser import Fuser


class AdaptiveSC(SeamCarving):
    """
    Adaptive Seam Carving implementation that uses feedback based on
    relative position changes to guide pixel removal decisions.

    Mathematical Framework:
    =====================

    1. Relative Position Calculation:
       For pixel at (i,j) in image of size (H,W):
       R(i,j) = (i/H, j/W) ∈ [0,1]²

    2. Position Displacement After Removal:
       When removing pixel at (r,c), for pixel at (i,j):
       - If i = r and j > c: new position = (i, j-1)
       - Relative displacement: δ = ||R_original - R_new||₂

    3. Feedback Function:
       F(removal_set) = Σ_{affected_pixels} δ + λ * Structure_Penalty

    4. Adaptive Cost Function:
       Cost(i,j) = E(i,j) + α(t) * F({(i,j)})
       where α(t) is time-adaptive feedback weight

    5. Adaptive Weight Update:
       α(t+1) = α(t) * β^sign(ΔCost_trend)
       where β > 1 for increasing costs, β < 1 for decreasing costs
    """

    def __init__(self, img, ratio, color=False, feedback_weight=0.3,
                 adaptation_rate=0.02):
        """
        Initialize Adaptive Seam Carving with Fuser energy processor

        Args:
            img: Input image (will be processed by Fuser)
            ratio: Target ratio for resizing
            color: Whether to visualize seams in color
            feedback_weight: Initial weight α₀ for feedback component
            adaptation_rate: Learning rate for weight adaptation
        """
        # Use Fuser class to generate sophisticated energy map
        energy_processor = Fuser(img)
        energy_map = energy_processor().image()

        self._color = color
        self._feedback_weight = feedback_weight
        self._initial_feedback_weight = feedback_weight
        self._adaptation_rate = adaptation_rate
        self._position_history = []
        self._removal_feedback = []
        self._cost_history = []

        super(AdaptiveSC, self).__init__(img.rgb(), energy_map, ratio)

    def _calculate_relative_positions(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Calculate normalized relative positions for all pixels.

        Mathematical formulation:
        For pixel at (i,j) in image of size (H,W):
        R(i,j) = (rᵢ, rⱼ) where:
        - rᵢ = i/H ∈ [0,1] (normalized row position)
        - rⱼ = j/W ∈ [0,1] (normalized column position)

        Args:
            image_shape: (height, width) of the image

        Returns:
            Array of shape (H,W,2) containing relative positions
        """
        height, width = image_shape
        positions = np.zeros((height, width, 2))

        # Vectorized calculation for efficiency
        row_indices, col_indices = np.meshgrid(
            np.arange(height), np.arange(width), indexing='ij'
        )

        positions[:, :, 0] = row_indices / height  # rᵢ = i/H
        positions[:, :, 1] = col_indices / width  # rⱼ = j/W

        return positions

    def _calculate_position_displacement(self, original_pos: np.ndarray,
                                         removal_row: int, removal_col: int) -> float:
        """
        Calculate position displacement after removing pixel at (removal_row, removal_col).

        Mathematical formulation:
        For pixel originally at (i,j) with relative position R₀(i,j) = (i/H, j/W):

        After removing pixel at (r,c):
        1. If i ≠ r: no displacement (different row)
        2. If i = r and j < c: no displacement (left of removal)
        3. If i = r and j > c: new position = (i, j-1)
           New relative position: R₁(i,j-1) = (i/H, (j-1)/(W-1))

        Displacement: δ(i,j) = ||R₀(i,j) - R₁(i,j-1)||₂
                             = |j/W - (j-1)/(W-1)|

        Total displacement: Δ = Σ_{affected pixels} δ(i,j)

        Args:
            original_pos: Original relative positions array
            removal_row, removal_col: Position of pixel to remove

        Returns:
            Total displacement score
        """
        height, width = original_pos.shape[:2]
        total_displacement = 0.0

        # Only pixels in the same row and to the right are affected
        for j in range(removal_col + 1, width):
            # Original relative position
            orig_rel_col = j / width

            # New relative position after removal (shifted left by 1)
            new_rel_col = (j - 1) / (width - 1)

            # Displacement in column direction
            displacement = abs(orig_rel_col - new_rel_col)
            total_displacement += displacement

        return total_displacement

    def _calculate_structure_penalty(self, image: np.ndarray, energy: np.ndarray,
                                     row: int, col: int) -> float:
        """
        Calculate penalty for disrupting local image structure.

        Mathematical formulation:
        Structure penalty considers local gradient disruption:

        S(i,j) = Σ_{(p,q)∈N(i,j)} w(p,q) * |E(p,q) - E(i,j)|

        where:
        - N(i,j) is the 8-neighborhood of (i,j)
        - w(p,q) is distance-weighted factor: w = 1/d(i,j,p,q)
        - E(i,j) is energy at position (i,j)

        Args:
            image: Current image
            energy: Current energy map
            row, col: Position to evaluate

        Returns:
            Structure penalty score
        """
        height, width = energy.shape
        penalty = 0.0
        center_energy = energy[row, col]

        # 8-neighborhood offsets
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for di, dj in neighbors:
            ni, nj = row + di, col + dj

            if 0 <= ni < height and 0 <= nj < width:
                # Distance weight: w = 1/d where d is Euclidean distance
                distance = math.sqrt(di * di + dj * dj)
                weight = 1.0 / distance

                # Energy difference
                energy_diff = abs(energy[ni, nj] - center_energy)

                penalty += weight * energy_diff

        return penalty

    def _evaluate_removal_candidate(self, image: np.ndarray, energy: np.ndarray,
                                    row: int, col: int) -> float:
        """
        Evaluate comprehensive cost of removing pixel at (row, col).

        Mathematical formulation:
        Total cost function:
        C(i,j,t) = E(i,j) + α(t) * [Δ(i,j) + λ * S(i,j)] + β * Smoothness(i,j)

        where:
        - E(i,j): Energy cost from Fuser processor
        - α(t): Time-adaptive feedback weight
        - Δ(i,j): Position displacement cost
        - S(i,j): Structure penalty
        - λ: Structure penalty weight
        - β: Smoothness penalty weight

        Args:
            image: Current image
            energy: Current energy map
            row, col: Position of pixel to evaluate

        Returns:
            Total cost score (lower is better)
        """
        # Base energy cost from Fuser processor
        energy_cost = energy[row, col]

        # Calculate position feedback
        original_positions = self._calculate_relative_positions(energy.shape)
        displacement_cost = self._calculate_position_displacement(
            original_positions, row, col
        )

        # Calculate structure penalty
        structure_penalty = self._calculate_structure_penalty(image, energy, row, col)

        # Combine costs with adaptive weighting
        λ = 0.1  # structure penalty weight
        total_cost = (energy_cost +
                      self._feedback_weight * (displacement_cost + λ * structure_penalty))

        return total_cost

    def _find_adaptive_seam(self, image: np.ndarray, energy: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find optimal seam using dynamic programming with adaptive feedback.

        Mathematical formulation:
        DP recurrence relation:
        DP(i,j) = C(i,j) + min{DP(i-1,j-1), DP(i-1,j), DP(i-1,j+1)} + Smoothness(i,j)

        Smoothness penalty:
        Smoothness(i,j) = β * |j - parent_j|

        where β penalizes large horizontal jumps between seam pixels.

        Args:
            image: Current image
            energy: Current energy map

        Returns:
            List of (row, col) indices forming optimal seam
        """
        height, width = energy.shape

        # DP table and parent tracking
        dp = np.full((height, width), np.inf)
        parent = np.full((height, width, 2), -1, dtype=int)

        # Initialize first row
        for j in range(width):
            dp[0, j] = self._evaluate_removal_candidate(image, energy, 0, j)

        # Fill DP table with adaptive cost evaluation
        for i in range(1, height):
            for j in range(width):
                candidate_cost = self._evaluate_removal_candidate(image, energy, i, j)

                # Check three possible parent positions
                for prev_j in [j - 1, j, j + 1]:
                    if 0 <= prev_j < width:
                        # Total cost = previous cost + current cost + smoothness penalty
                        smoothness_penalty = 0.05 * abs(j - prev_j)  # β = 0.05
                        total_cost = dp[i - 1, prev_j] + candidate_cost + smoothness_penalty

                        if total_cost < dp[i, j]:
                            dp[i, j] = total_cost
                            parent[i, j] = [i - 1, prev_j]

        # Backtrack to find optimal seam
        seam = []
        min_col = np.argmin(dp[height - 1])
        current_pos = [height - 1, min_col]

        while current_pos[0] >= 0:
            seam.append((current_pos[0], current_pos[1]))
            if current_pos[0] == 0:
                break
            current_pos = parent[current_pos[0], current_pos[1]]

        seam.reverse()
        return seam

    def _update_adaptive_weight(self, current_cost: float):
        """
        Update feedback weight based on cost trend analysis.

        Mathematical formulation:
        Weight adaptation rule:
        α(t+1) = α(t) * exp(γ * sign(ΔC̄))

        where:
        - ΔC̄ = moving average of cost differences
        - γ = adaptation rate
        - sign(x) = 1 if x > 0, -1 if x < 0, 0 if x = 0

        Intuition: Increase weight when costs are rising (poor performance),
                  decrease when costs are falling (good performance)

        Args:
            current_cost: Cost of the current seam removal
        """
        self._cost_history.append(current_cost)

        # Need at least 5 samples for trend analysis
        if len(self._cost_history) >= 5:
            # Calculate moving average of cost differences
            recent_costs = self._cost_history[-5:]
            cost_diffs = np.diff(recent_costs)
            avg_cost_trend = np.mean(cost_diffs)

            # Adaptive weight update
            γ = self._adaptation_rate  # adaptation rate
            if avg_cost_trend > 0:
                # Costs increasing - increase feedback weight
                self._feedback_weight *= (1 + γ)
            elif avg_cost_trend < 0:
                # Costs decreasing - decrease feedback weight
                self._feedback_weight *= (1 - γ)

            # Clamp weight to reasonable bounds
            self._feedback_weight = np.clip(self._feedback_weight, 0.05, 1.0)

    def _remove_seam(self, image: np.ndarray, energy: np.ndarray,
                     seam: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove seam from image and energy map, updating feedback history.

        Args:
            image: Current image
            energy: Current energy map
            seam: List of (row, col) indices to remove

        Returns:
            Tuple of (new_image, new_energy)
        """
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        if self._color:
            # Visualize seam without actually removing pixels
            for row, col in seam:
                image[row, col] = [0, 255, 0]
            return image, energy

        # Actually remove seam pixels
        new_image = np.zeros((height, width - 1, channels), dtype=image.dtype)
        new_energy = np.zeros((height, width - 1), dtype=energy.dtype)

        seam_dict = {row: col for row, col in seam}

        for i in range(height):
            remove_col = seam_dict.get(i, -1)

            # Create new row without the seam pixel
            if remove_col >= 0:
                new_image[i] = np.concatenate([
                    image[i, :remove_col],
                    image[i, remove_col + 1:]
                ])
                new_energy[i] = np.concatenate([
                    energy[i, :remove_col],
                    energy[i, remove_col + 1:]
                ])
            else:
                new_image[i] = image[i, :-1]  # Remove last pixel if no seam pixel
                new_energy[i] = energy[i, :-1]

        return new_image, new_energy

    @Decorators.Loggers.log_class_method_time
    def _main(self, *args, **kwargs):
        """
        Main adaptive seam carving algorithm with mathematical optimization.

        Algorithm Overview:
        1. Initialize with Fuser-generated energy map
        2. For each seam removal iteration:
           a. Find optimal seam using adaptive DP
           b. Calculate removal cost
           c. Update adaptive feedback weight
           d. Remove seam and update structures
        3. Return final retargeted image
        """
        image = self._img.copy()
        energy = self._energy.copy()

        height, width, channels = image.shape
        target_width = int(width * self._ratio)
        seams_to_remove = width - target_width

        print(f"Adaptive SC: Removing {seams_to_remove} seams from {width}x{height} image")
        print(f"Initial feedback weight α₀ = {self._feedback_weight:.3f}")

        for iteration in range(seams_to_remove):
            if iteration % max(1, seams_to_remove // 10) == 0:
                print(f"Processing seam {iteration + 1}/{seams_to_remove}, α = {self._feedback_weight:.3f}")

            # Find optimal seam using adaptive feedback
            seam = self._find_adaptive_seam(image, energy)

            # Calculate total cost for this seam
            total_cost = sum(energy[row, col] for row, col in seam)

            # Update adaptive feedback weight based on performance
            self._update_adaptive_weight(total_cost)

            # Store removal feedback for analysis
            self._removal_feedback.append({
                'seam': seam,
                'cost': total_cost,
                'iteration': iteration,
                'feedback_weight': self._feedback_weight
            })

            # Remove seam from image and energy map
            image, energy = self._remove_seam(image, energy, seam)

        print(f"Adaptive SC completed.")
        print(f"Weight evolution: {self._initial_feedback_weight:.3f} → {self._feedback_weight:.3f}")

        if self._cost_history:
            print(f"Cost statistics: mean={np.mean(self._cost_history):.3f}, "
                  f"std={np.std(self._cost_history):.3f}")

        return image