import numpy as np
import cv2
import time
from typing import List, Tuple, Optional


class SpatialRemover:
    def __init__(self, array: np.ndarray):
        self.array = array.copy().tolist()
        self.original_array = array.copy().tolist()
        self.landmarks = []
        self._find_landmarks()

    def _distance(self, i1: int, i2: int) -> float:
        return abs(i1 - i2)

    def _find_landmarks(self, percentile: float = 0.8) -> None:
        start_time = time.perf_counter()

        all_points = [(i, val) for i, val in enumerate(self.array)]
        all_points.sort(key=lambda x: x[1], reverse=True)

        landmark_count = max(1, int(len(all_points) * (1.0 - percentile)))
        self.landmarks = all_points[:landmark_count]

        end_time = time.perf_counter()
        print(f"_find_landmarks took: {(end_time - start_time) * 1000:.3f} ms")

    def remove_next(self) -> Optional[Tuple[int, float]]:
        start_time = time.perf_counter()

        if not self.array:
            return None

        min_index = 0
        min_value = self.array[0]
        for i in range(1, len(self.array)):
            if self.array[i] < min_value:
                min_value = self.array[i]
                min_index = i

        to_remove = (min_index, min_value)

        for offset in [-1, 1]:
            adj_index = min_index + offset
            if 0 <= adj_index < len(self.array):
                self.array[adj_index] *= 0.95

        self.array.pop(min_index)

        for i, (landmark_idx, landmark_val) in enumerate(self.landmarks):
            if landmark_idx > min_index:
                self.landmarks[i] = (landmark_idx - 1, landmark_val)

        end_time = time.perf_counter()
        print(f"remove_next took: {(end_time - start_time) * 1000:.3f} ms")

        return to_remove

    def visualize(self, title: str, width: int = 800, height: int = 200) -> np.ndarray:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        if not self.array:
            return img

        max_val = max(self.array)
        min_val = min(self.array)

        bar_width = width // len(self.array)

        landmark_indices = {idx for idx, _ in self.landmarks}

        for i, val in enumerate(self.array):
            normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            bar_height = int(normalized * (height - 40))

            if i in landmark_indices:
                color = (0, 255, 255)  # Yellow for landmarks
            else:
                color = (int(255 * normalized), int(255 * normalized), int(255 * (1 - normalized)))

            cv2.rectangle(img,
                          (i * bar_width, height - bar_height - 20),
                          ((i + 1) * bar_width - 1, height - 20),
                          color, -1)

        cv2.putText(img, f"{title} (Size: {len(self.array)})",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return img

    def visualize_original(self, title: str, width: int = 800, height: int = 200) -> np.ndarray:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        if not self.original_array:
            return img

        max_val = max(self.original_array)
        min_val = min(self.original_array)

        bar_width = width // len(self.original_array)

        for i, val in enumerate(self.original_array):
            normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            bar_height = int(normalized * (height - 40))

            color = (int(255 * normalized), int(255 * normalized), int(255 * (1 - normalized)))

            cv2.rectangle(img,
                          (i * bar_width, height - bar_height - 20),
                          ((i + 1) * bar_width - 1, height - 20),
                          color, -1)

        cv2.putText(img, f"{title} (Size: {len(self.original_array)})",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return img

    def size(self) -> int:
        return len(self.array)


def main():
    np.random.seed(42)
    size = 50
    array = np.random.uniform(0, 100, size)

    total_start = time.perf_counter()

    remover = SpatialRemover(array)

    original_viz = remover.visualize_original("Original Array")
    cv2.imshow("Original", original_viz)

    for i in range(10):
        removed = remover.remove_next()
        if removed:
            index, value = removed
            print(f"Removed: index={index} value={value:.2f}")

            current_viz = remover.visualize(f"After {i + 1} removals")
            cv2.imshow("Current", current_viz)
            cv2.waitKey(500)
        else:
            print("No more elements to remove")
            break

    final_viz = remover.visualize("Final Array")
    cv2.imshow("Final", final_viz)

    total_end = time.perf_counter()
    print(f"Total execution time: {(total_end - total_start) * 1000:.3f} ms")

    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()