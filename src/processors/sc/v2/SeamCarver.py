import numpy as np
import cv2
import time
from typing import List, Tuple, Optional
from config.decorators import Decorators


class SeamCarver:
    def __init__(self, image: np.ndarray, importanceMap: np.ndarray):
        self.orig_img = image.copy()
        self.img = image.copy()
        self.imp_map = importanceMap.copy()
        self.h, self.w = image.shape[:2]

        self.fwd_energy = np.zeros((self.h, self.w), dtype=np.float32)
        self.valid = np.zeros((self.h, self.w), dtype=bool)
        self.h_grad = np.zeros((self.h, self.w), dtype=np.float32)
        self.grad_valid = np.zeros((self.h, self.w), dtype=bool)

        max_ch = self.img.shape[2] if len(self.img.shape) > 2 else 1
        self.temp_img = np.zeros((self.h, self.w, max_ch), dtype=self.img.dtype)
        self.temp_imp = np.zeros((self.h, self.w), dtype=np.float32)
        self.temp_energy = np.zeros((self.h, self.w), dtype=np.float32)
        self.temp_valid = np.zeros((self.h, self.w), dtype=bool)

        self.cost_l = np.full((self.h, self.w), np.inf, dtype=np.float32)
        self.cost_u = np.full((self.h, self.w), np.inf, dtype=np.float32)
        self.cost_r = np.full((self.h, self.w), np.inf, dtype=np.float32)

    def computeHGrad(self, cols: Optional[np.ndarray] = None):
        if cols is None:
            self.h_grad[:, 1:-1] = np.abs(self.imp_map[:, 2:] - self.imp_map[:, :-2])
            self.h_grad[:, 0] = 0
            self.h_grad[:, -1] = 0
            self.grad_valid.fill(True)
        else:
            for j in cols:
                if 1 <= j < self.w - 1:
                    self.h_grad[:, j] = np.abs(self.imp_map[:, j + 1] - self.imp_map[:, j - 1])
                    self.grad_valid[:, j] = True

    @Decorators.Loggers.log_class_method_time
    def calcFwdEnergy(self, left: int = None, right: int = None):
        if left is not None or right is not None:
            self.initNeighbors(left, right)

        affected = np.where(~self.grad_valid.any(axis=0))[0]
        if len(affected) > 0:
            self.computeHGrad(affected)

        mask = ~self.valid[0, :]
        self.fwd_energy[0, mask] = self.imp_map[0, mask]
        self.valid[0, mask] = True

        for i in range(1, self.h):
            recalc = ~self.valid[i, :]
            if not recalc.any():
                continue

            old_vals = self.fwd_energy[i, :].copy()

            self.cost_l[:, :self.w].fill(np.inf)
            self.cost_u[:, :self.w].fill(np.inf)
            self.cost_r[:, :self.w].fill(np.inf)

            left_ok = (np.arange(self.w) > 0) & recalc
            if left_ok.any():
                idx = np.where(left_ok)[0]
                cuts = np.abs(self.imp_map[i - 1, idx] - self.imp_map[i, idx - 1])
                self.cost_l[i, idx] = (self.h_grad[i, idx] + cuts + self.fwd_energy[i - 1, idx - 1])

            up_ok = ((np.arange(self.w) > 0) & (np.arange(self.w) < self.w - 1) & recalc)
            if up_ok.any():
                idx = np.where(up_ok)[0]
                self.cost_u[i, idx] = (self.h_grad[i, idx] + self.fwd_energy[i - 1, idx])

            right_ok = (np.arange(self.w) < self.w - 1) & recalc
            if right_ok.any():
                idx = np.where(right_ok)[0]
                cuts = np.abs(self.imp_map[i - 1, idx] - self.imp_map[i, idx + 1])
                self.cost_r[i, idx] = (self.h_grad[i, idx] + cuts + self.fwd_energy[i - 1, idx + 1])

            min_costs = np.minimum(np.minimum(self.cost_l[i, :self.w], self.cost_u[i, :self.w]),
                                   self.cost_r[i, :self.w])

            self.fwd_energy[i, recalc] = (self.imp_map[i, recalc] + min_costs[recalc])
            self.valid[i, recalc] = True

            if i < self.h - 1:
                changes = np.abs(self.fwd_energy[i, :] - old_vals)
                rel_changes = changes / (np.abs(old_vals) + 1e-10)
                sig_changes = (rel_changes > 0.001) & recalc

                if sig_changes.any():
                    change_idx = np.where(sig_changes)[0]
                    for j in change_idx:
                        start = max(0, j - 1)
                        end = min(self.w, j + 2)
                        self.valid[i + 1, start:end] = False

    def initNeighbors(self, left: int = None, right: int = None):
        neighbors = set()
        if left is not None:
            neighbors.update([left])
        if right is not None:
            neighbors.update([right])

        valid_n = [j for j in neighbors if 0 <= j < self.w]
        if valid_n:
            self.valid[:, valid_n] = False
            self.grad_valid[:, valid_n] = False

    def findStart(self) -> int:
        return np.argmin(self.fwd_energy[-1, :])

    def selectDir(self, curr_j: int, i: int) -> int:
        candidates = []
        if curr_j > 0:
            candidates.append((curr_j - 1, self.fwd_energy[i, curr_j - 1]))
        candidates.append((curr_j, self.fwd_energy[i, curr_j]))
        if curr_j < self.w - 1:
            candidates.append((curr_j + 1, self.fwd_energy[i, curr_j + 1]))

        return min(candidates, key=lambda x: x[1])[0]

    @Decorators.Loggers.log_class_method_time
    def removeSeam(self, start_j: int) -> Tuple[int, int]:
        seam = np.zeros(self.h, dtype=np.int32)
        seam[-1] = start_j
        curr_j = start_j

        for i in range(self.h - 2, -1, -1):
            curr_j = self.selectDir(curr_j, i)
            seam[i] = curr_j

        left_n = None
        right_n = None

        for i in range(self.h):
            sj = seam[i]

            if sj < self.w - 1:
                self.temp_img[i, :sj] = self.img[i, :sj]
                self.temp_img[i, sj:self.w - 1] = self.img[i, sj + 1:]

                self.temp_imp[i, :sj] = self.imp_map[i, :sj]
                self.temp_imp[i, sj:self.w - 1] = self.imp_map[i, sj + 1:]

                self.temp_energy[i, :sj] = self.fwd_energy[i, :sj]
                self.temp_energy[i, sj:self.w - 1] = self.fwd_energy[i, sj + 1:]

                self.temp_valid[i, :sj] = self.valid[i, :sj]
                self.temp_valid[i, sj:self.w - 1] = self.valid[i, sj + 1:]
            else:
                self.temp_img[i, :sj] = self.img[i, :sj]
                self.temp_imp[i, :sj] = self.imp_map[i, :sj]
                self.temp_energy[i, :sj] = self.fwd_energy[i, :sj]
                self.temp_valid[i, :sj] = self.valid[i, :sj]

            if i == 0:
                left_n = sj - 1 if sj > 0 else None
                right_n = sj if sj < self.w - 1 else None

        new_w = self.w - 1
        self.img, self.temp_img = self.temp_img[:, :new_w], self.img
        self.imp_map, self.temp_imp = self.temp_imp[:, :new_w], self.imp_map
        self.fwd_energy, self.temp_energy = self.temp_energy[:, :new_w], self.fwd_energy
        self.valid, self.temp_valid = self.temp_valid[:, :new_w], self.valid

        self.h_grad = self.h_grad[:, :new_w]
        self.grad_valid = self.grad_valid[:, :new_w]
        self.cost_l = self.cost_l[:, :new_w]
        self.cost_u = self.cost_u[:, :new_w]
        self.cost_r = self.cost_r[:, :new_w]
        self.w = new_w

        return left_n, right_n

    @Decorators.Loggers.log_class_method_time
    def seamCarve(self, ratio: float) -> np.ndarray:
        if ratio >= 1.0:
            print("Target ratio >= 1.0, returning original image")
            return self.img

        target_w = int(self.orig_img.shape[1] * ratio)
        to_remove = self.w - target_w

        if to_remove <= 0:
            print("No seams to remove")
            return self.img

        self.calcFwdEnergy()

        for n in range(to_remove):
            start = time.perf_counter()
            start_j = self.findStart()
            left_n, right_n = self.removeSeam(start_j)

            if n < to_remove - 1:
                self.calcFwdEnergy(left_n, right_n)

            end = time.perf_counter()
            print(f"Seam {n + 1}/{to_remove} removed in {(end - start) * 1000:.2f}ms")

            if n % 10 == 0:
                print(f"Progress: {n}/{to_remove} seams removed")

        return self.img

    def visualize(self, seam: List[int] = None) -> np.ndarray:
        viz = self.img.copy()
        if seam:
            for i, j in enumerate(seam):
                if j < self.w:
                    viz[i, j] = [0, 0, 255]
        return viz