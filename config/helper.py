import io

import PIL.Image
import numpy as np


class Helper:
    class Image:
        """
        Used to convert rgb pixel to grayscale
        """
        @staticmethod
        def NTSC_formula(r, g, b) -> float:
            return 0.299 * r + 0.587 * g + 0.114 * b

        @staticmethod
        def get_stream(img, mimetype):
            image = PIL.Image.fromarray(img)

            stream = io.BytesIO()

            image.save(stream, format=mimetype.split('/')[1])

            return stream.getvalue()

        @staticmethod
        def normalize_map(input_map):
            min_val = np.min(input_map)
            max_val = np.max(input_map)

            # Prevent division by zero in case all values are the same
            if min_val == max_val:
                return np.zeros_like(input_map)

            normalized_map = (input_map - min_val) / (max_val - min_val)

            return normalized_map
        
        @staticmethod
        def forward_energy(matrix, energy, width, j, i, step):
            left_bound = max(i - 1, 0)
            right_bound = min(i + 1, width - 1)

            cu_val = abs(energy[j - step, right_bound] - energy[j, left_bound])

            cl_val = abs(energy[j - step, i] - energy[j, left_bound]) + cu_val
            cr_val = abs(energy[j - step, i] - energy[j, right_bound]) + cu_val

            return min(
                cl_val + matrix[j - step, left_bound],
                cu_val + matrix[j - step, i],
                cr_val + matrix[j - step, right_bound],
            )
        
        @staticmethod
        def forward_energy_3d(matrix, matrix_old, energy, width, j, i, step):
            left_bound = max(i - 1, 0)
            right_bound = min(i + 1, width - 1)

            A = energy[j - step, left_bound]
            B = energy[j - step, i]
            C = energy[j - step, right_bound]
            D = energy[j, left_bound]
            F = energy[j, right_bound]

            a = matrix_old[j - step, left_bound]
            b = matrix_old[j - step, i]
            e = matrix_old[j, i]

            Cb = abs(C - b)

            Fe = abs(F - e)
            FD = abs(F - D)

            # r = Fe + FD

            # cl = r + Cb + abs(B - D) + abs(B - a)
            # cu = r + Cb + abs(A - C)
            # cr = r + abs(B - F)

            cl = FD + Cb + Fe + abs(D - B) + abs(B - a)
            cm = FD + Cb + Fe + abs(A - C)
            cr = FD + Fe + abs(F - B)

            return min(
                cl + matrix[j - step, left_bound],
                cm + matrix[j - step, i],
                cr + matrix[j - step, right_bound],
            )

    class Lists:
        @staticmethod
        def sort_with_indices(lst):
            sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
            sorted_indices = [index for index, _ in sorted_lst]
            return sorted_lst, sorted_indices

    class Math:
        @staticmethod
        def sigmoid(x):
            return 1 / 1 + np.exp(x)

        @staticmethod
        def scale(x, start=0.0, end=255.0):
            return (x - start) / (end - start)

        @staticmethod
        def get_min(matrix, j, k, start, end):
            l = max(k - 1, start)
            r = min(k + 1, end)

            left = matrix[j][l]
            mid = matrix[j][k]
            right = matrix[j][r]

            if left <= mid and left <= right:
                temp = left, l
            elif mid <= left and mid <= right:
                temp = mid, k
            else:
                temp = right, r

            return temp
