import numpy as np
import cv2
import time
from typing import List, Tuple, Optional
from collections import deque

from config.decorators import Decorators


class ProgressiveSeamCarver:
    def __init__(self, image: np.ndarray, importanceMap: np.ndarray, direction_memory_size: int = 5):
        self.originalImage = image.copy()
        self.currentImage = image.copy()
        self.importanceMap = importanceMap.copy()
        self.height, self.width = image.shape[:2]
        self.forwardEnergyMap = np.zeros((self.height, self.width))
        self.energyValid = np.zeros((self.height, self.width), dtype=bool)
        self.direction_memory_size = direction_memory_size

        self.edge_width = self.width // 4
        self.seams_removed_left = 0
        self.seams_removed_right = 0

    def _is_in_edge_zones(self, seam):
        left_count = sum(1 for col in seam if col < self.edge_width)
        right_count = sum(1 for col in seam if col >= self.width - self.edge_width)
        center_count = len(seam) - left_count - right_count

        if left_count > center_count and left_count > right_count:
            return 'left'
        elif right_count > center_count and right_count > left_count:
            return 'right'
        else:
            return 'center'

    def _find_best_edge_seam(self):
        last_row = self.forwardEnergyMap[self.height - 1, :]
        sorted_indices = np.argsort(last_row)

        for start_j in sorted_indices:
            seam = self._trace_seam(start_j)
            zone = self._is_in_edge_zones(seam)

            balance_diff = abs(self.seams_removed_left - self.seams_removed_right)

            if zone == 'left' and (balance_diff < 10 or self.seams_removed_left <= self.seams_removed_right):
                return start_j, 'left'
            elif zone == 'right' and (balance_diff < 10 or self.seams_removed_right <= self.seams_removed_left):
                return start_j, 'right'

        for start_j in sorted_indices:
            seam = self._trace_seam(start_j)
            zone = self._is_in_edge_zones(seam)
            if zone in ['left', 'right']:
                return start_j, zone

        return sorted_indices[0], 'center'

    def _trace_seam(self, startJ):
        seam = [startJ]
        currentJ = startJ

        for i in range(self.height - 2, -1, -1):
            candidates = []
            for nextJ in range(max(0, currentJ - 1), min(self.width, currentJ + 2)):
                candidates.append((nextJ, self.forwardEnergyMap[i, nextJ]))

            bestJ = min(candidates, key=lambda x: x[1])[0]
            seam.append(bestJ)
            currentJ = bestJ

        seam.reverse()
        return seam

    @Decorators.Loggers.log_class_method_time
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

    @Decorators.Loggers.log_class_method_time
    def removeSeamDirect(self, startJ, zone):
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

        if zone == 'left':
            self.seams_removed_left += 1
        elif zone == 'right':
            self.seams_removed_right += 1

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

        self.currentImage = newImage
        self.importanceMap = newImportanceMap
        self.forwardEnergyMap = newForwardEnergyMap
        self.energyValid = newEnergyValid
        self.width -= 1

        if zone in ['left', 'right']:
            self.edge_width = self.width // 4

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

        self.calculateForwardEnergy()

        for seamNum in range(seamsToRemove):
            startJ, zone = self._find_best_edge_seam()
            leftNeighbor, rightNeighbor = self.removeSeamDirect(startJ, zone)

            if seamNum < seamsToRemove - 1:
                self.calculateForwardEnergy(leftNeighbor, rightNeighbor)

        return self.currentImage