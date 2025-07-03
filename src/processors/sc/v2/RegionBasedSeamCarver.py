import numpy as np
import cv2
import time
from typing import List, Tuple, Optional
from collections import deque
from scipy.ndimage import maximum_filter

from config.decorators import Decorators


class RegionBasedSeamCarver:
    def __init__(self, image: np.ndarray, importanceMap: np.ndarray, direction_memory_size: int = 5,
                 num_regions: int = 5):
        self.originalImage = image.copy()
        self.currentImage = image.copy()
        self.importanceMap = importanceMap.copy()
        self.height, self.width = image.shape[:2]
        self.forwardEnergyMap = np.zeros((self.height, self.width))
        self.energyValid = np.zeros((self.height, self.width), dtype=bool)
        self.direction_memory_size = direction_memory_size

        self.regions = self._setup_regions(num_regions)
        self.region_quotas = []
        self.region_removed = []
        self._cached_seams = {}
        self._last_energy_hash = None

    def _setup_regions(self, num_regions):
        local_max = maximum_filter(self.importanceMap, size=50) == self.importanceMap
        maxima_coords = np.where(local_max & (self.importanceMap > np.percentile(self.importanceMap, 80)))
        maxima_x = sorted(list(set(maxima_coords[1])))

        if len(maxima_x) > 5:
            step = len(maxima_x) // 3
            maxima_x = [maxima_x[0], maxima_x[step], maxima_x[2 * step]]

        if len(maxima_x) >= 2:
            regions = [0]
            for i in range(len(maxima_x) - 1):
                regions.append((maxima_x[i] + maxima_x[i + 1]) // 2)
            regions.append(self.width)
        else:
            regions = [0, self.width // 5, 2 * self.width // 5, self.width]

        return regions

    def _distribute_quotas(self, total_seams):
        region_scores = []
        for i in range(len(self.regions) - 1):
            start_x, end_x = self.regions[i], self.regions[i + 1]
            region_scores.append(np.mean(self.importanceMap[:, start_x:end_x]))

        max_importance = max(region_scores)
        removability = [max_importance - score for score in region_scores]
        total_removability = sum(removability)

        if total_removability == 0:
            quotas = [total_seams // len(region_scores)] * len(region_scores)
        else:
            quotas = [int(total_seams * (score / total_removability)) for score in removability]

        remaining = total_seams - sum(quotas)
        for i in range(remaining):
            quotas[i % len(quotas)] += 1

        self.region_quotas = quotas
        self.region_removed = [0] * len(quotas)

    def _get_seam_region(self, seam):
        region_counts = np.zeros(len(self.regions) - 1)
        for col_idx in seam:
            for region_idx in range(len(self.regions) - 1):
                if self.regions[region_idx] <= col_idx < self.regions[region_idx + 1]:
                    region_counts[region_idx] += 1
                    break
        return np.argmax(region_counts)

    def _find_best_seam(self):
        energy_hash = hash(self.forwardEnergyMap.tobytes())
        if energy_hash != self._last_energy_hash:
            self._cached_seams.clear()
            self._last_energy_hash = energy_hash

        last_row = self.forwardEnergyMap[self.height - 1, :]
        sorted_indices = np.argsort(last_row)

        for start_j in sorted_indices:
            if start_j in self._cached_seams:
                seam, region = self._cached_seams[start_j]
            else:
                seam = self._trace_seam(start_j)
                region = self._get_seam_region(seam)
                self._cached_seams[start_j] = (seam, region)

            if self.region_removed[region] < self.region_quotas[region]:
                return start_j, region

        remaining = [q - r for q, r in zip(self.region_quotas, self.region_removed)]
        best_region = np.argmax(remaining)

        for start_j in sorted_indices:
            if start_j in self._cached_seams:
                seam, region = self._cached_seams[start_j]
            else:
                seam = self._trace_seam(start_j)
                region = self._get_seam_region(seam)
                self._cached_seams[start_j] = (seam, region)

            if region == best_region:
                return start_j, region

        return sorted_indices[0], self._get_seam_region(self._trace_seam(sorted_indices[0]))

    def _trace_seam(self, startJ):
        seam = [startJ]
        currentJ = startJ

        for i in range(self.height - 2, -1, -1):
            candidates = []
            for nextJ in [currentJ - 1, currentJ, currentJ + 1]:
                if 0 <= nextJ < self.width:
                    candidates.append((nextJ, self.forwardEnergyMap[i, nextJ]))

            bestJ = min(candidates, key=lambda x: x[1])[0]
            seam.append(bestJ)
            currentJ = bestJ

        seam.reverse()
        return seam

    # @Decorators.Loggers.log_class_method_time
    def calculateForwardEnergy(self, leftNeighbor: int = None, rightNeighbor: int = None):
        if leftNeighbor is not None:
            for nextJ in range(max(0, leftNeighbor - 1), min(self.width, leftNeighbor + 2)):
                self.energyValid[0, nextJ] = False

        if rightNeighbor is not None:
            for nextJ in range(max(0, rightNeighbor - 1), min(self.width, rightNeighbor + 2)):
                self.energyValid[0, nextJ] = False

        for i in range(self.height):
            for j in range(self.width):
                if self.energyValid[i, j]:
                    continue

                energy = self.importanceMap[i, j]

                if i == 0:
                    self.forwardEnergyMap[i, j] = energy
                else:
                    costs = []
                    for prevJ in range(max(0, j - 1), min(self.width, j + 2)):
                        if prevJ == j - 1:
                            cost = self._getCL(i, j, prevJ) + self.forwardEnergyMap[i - 1, prevJ]
                        elif prevJ == j:
                            cost = self._getCU(i, j, prevJ) + self.forwardEnergyMap[i - 1, prevJ]
                        else:
                            cost = self._getCR(i, j, prevJ) + self.forwardEnergyMap[i - 1, prevJ]
                        costs.append(cost)

                    self.forwardEnergyMap[i, j] = energy + min(costs)

                self.energyValid[i, j] = True

    def _getCL(self, i, j, prevJ):
        if i == 0 or j <= 0 or j >= self.width - 1:
            return 0
        return (abs(self.importanceMap[i - 1, j] - self.importanceMap[i, j - 1]) +
                abs(self.importanceMap[i - 1, j + 1] - self.importanceMap[i, j - 1]) +
                abs(self.importanceMap[i - 1, prevJ] - self.importanceMap[i, j + 1]))

    def _getCU(self, i, j, prevJ):
        if i == 0 or j <= 0 or j >= self.width - 1:
            return 0
        return (abs(self.importanceMap[i - 1, j + 1] - self.importanceMap[i, j - 1]) +
                abs(self.importanceMap[i - 1, prevJ] - self.importanceMap[i, j + 1]))

    def _getCR(self, i, j, prevJ):
        if i == 0 or j <= 0 or j >= self.width - 1:
            return 0
        return (abs(self.importanceMap[i - 1, j] - self.importanceMap[i, j + 1]) +
                abs(self.importanceMap[i - 1, j + 1] - self.importanceMap[i, j - 1]) +
                abs(self.importanceMap[i - 1, prevJ] - self.importanceMap[i, j + 1]))

    def _selectDirection(self, currentJ, candidates, direction_history):
        if len(candidates) == 1:
            return candidates[0][0]

        minCost = min(cost for _, cost in candidates)
        equal_candidates = [(j, cost) for j, cost in candidates if abs(cost - minCost) < 1e-6]

        if len(equal_candidates) == 1:
            return equal_candidates[0][0]

        if len(direction_history) == 0:
            return currentJ if any(j == currentJ for j, _ in equal_candidates) else equal_candidates[0][0]

        direction_score = sum(direction_history) / len(direction_history)

        for j, _ in equal_candidates:
            if j == currentJ:
                return j
            elif j == currentJ - 1 and direction_score < -0.2:
                return j
            elif j == currentJ + 1 and direction_score > 0.2:
                return j

        return equal_candidates[0][0]

    # @Decorators.Loggers.log_class_method_time
    def removeSeamDirect(self, startJ, region_idx):
        seam = [startJ]
        currentJ = startJ
        direction_history = deque(maxlen=self.direction_memory_size)

        for i in range(self.height - 2, -1, -1):
            candidates = [(nextJ, self.forwardEnergyMap[i, nextJ])
                          for nextJ in range(max(0, currentJ - 1), min(self.width, currentJ + 2))]

            bestJ = self._selectDirection(currentJ, candidates, direction_history)
            direction_history.append(bestJ - currentJ)
            seam.append(bestJ)
            currentJ = bestJ

        seam.reverse()
        self.region_removed[region_idx] += 1

        newImage = np.zeros((self.height, self.width - 1, self.currentImage.shape[2]), dtype=self.currentImage.dtype)
        newImportanceMap = np.zeros((self.height, self.width - 1))
        newForwardEnergyMap = np.zeros((self.height, self.width - 1))
        newEnergyValid = np.zeros((self.height, self.width - 1), dtype=bool)

        for i in range(self.height):
            seamJ = seam[i]
            newImage[i, :seamJ] = self.currentImage[i, :seamJ]
            newImage[i, seamJ:] = self.currentImage[i, seamJ + 1:]
            newImportanceMap[i, :seamJ] = self.importanceMap[i, :seamJ]
            newImportanceMap[i, seamJ:] = self.importanceMap[i, seamJ + 1:]
            newForwardEnergyMap[i, :seamJ] = self.forwardEnergyMap[i, :seamJ]
            newForwardEnergyMap[i, seamJ:] = self.forwardEnergyMap[i, seamJ + 1:]
            newEnergyValid[i, :seamJ] = self.energyValid[i, :seamJ]
            newEnergyValid[i, seamJ:] = self.energyValid[i, seamJ + 1:]

        for i in range(len(self.regions)):
            if self.regions[i] > seam[0]:
                self.regions[i] -= 1

        self.currentImage = newImage
        self.importanceMap = newImportanceMap
        self.forwardEnergyMap = newForwardEnergyMap
        self.energyValid = newEnergyValid
        self.width -= 1
        self._cached_seams.clear()

        return (seam[0] - 1 if seam[0] > 0 else None,
                seam[0] if seam[0] < self.width else None)

    @Decorators.Loggers.log_class_method_time
    def seamCarve(self, targetRatio: float):
        if targetRatio >= 1.0:
            return self.currentImage

        targetWidth = int(self.originalImage.shape[1] * targetRatio)
        seamsToRemove = self.width - targetWidth

        if seamsToRemove <= 0:
            return self.currentImage

        self._distribute_quotas(seamsToRemove)
        self.calculateForwardEnergy()

        for seamNum in range(seamsToRemove):
            startJ, region_idx = self._find_best_seam()
            leftNeighbor, rightNeighbor = self.removeSeamDirect(startJ, region_idx)

            if seamNum < seamsToRemove - 1:
                self.calculateForwardEnergy(leftNeighbor, rightNeighbor)

        return self.currentImage