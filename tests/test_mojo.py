import math
import time

import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2RGB, COLOR_BGR2GRAY, Sobel, CV_64F, normalize, CV_8U, NORM_MINMAX

from config.plotter import Plotter


def sobel(gray, k=3):
    gradient_x = Sobel(gray, CV_64F, 1, 0, ksize=k)
    gradient_y = Sobel(gray, CV_64F, 0, 1, ksize=k)

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    gradient_normalized = normalize(gradient_magnitude, None, 0, 255, NORM_MINMAX, CV_8U)

    return gradient_normalized


def get_min(matrix, j, k, start, end):
    l = max(k - 1, start)
    r = min(k + 1, end)

    try:
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
    except Exception as e:
        print(l, k, r, start, end)
        raise e


def topdown(energy):
    height, width = energy.shape
    
    matrix = np.zeros_like(energy)

    for j in range(0, height, 1):
        for i in range(width):
            if j == 0:
                matrix[j, i] = energy[j, i]
            else:
                left_bound = max(i - 1, 0)
                right_bound = min(i + 1, width - 1)

                cu_val = abs(energy[j - 1, right_bound] - energy[j, left_bound])

                cl_val = abs(energy[j - 1, i] - energy[j, left_bound]) + cu_val
                cr_val = abs(energy[j - 1, i] - energy[j, right_bound]) + cu_val

                # print(cl_val, cu_val, cr_val)

                matrix[j, i] = energy[j, i] + min(
                    cl_val + matrix[j - 1, left_bound],
                    cu_val + matrix[j - 1, i],
                    cr_val + matrix[j - 1, right_bound],
                )
                
    return matrix


def bottomup(rgb, energy, ratio):
    h, w, z = rgb.shape

    n = int(w - w * ratio)

    old_image = rgb.copy()
    old_matrix = energy.copy()

    new_image = [[] for i in range(h)]
    new_matrix = [[] for i in range(h)]

    row = old_matrix[-1]

    for k in range(n):
        i = np.argmin(row)

        row = np.delete(row, i, axis=0)

        for j in range(h - 1, -1, -1):
            v, i = get_min(old_matrix, j, i, 0, len(old_matrix[j]) - 1)

            new_image[j] = np.delete(old_image[j], i, axis=0)
            new_matrix[j] = np.delete(old_matrix[j], i, axis=0)

            # old_image[j][i] = [0, 255, 0]
            # new_image[j] = old_image[j]
            #
            # old_matrix[j][i] = 8000
            # new_matrix[j] = old_matrix[j]

        old_image = new_image
        old_matrix = new_matrix

    return np.array(new_image)


def SeamCarving(rgb, gray, ratio):
    start_time = time.perf_counter()
    energy = topdown(gray)
    end_time = time.perf_counter()

    execution_time = end_time - start_time

    print("topdown time:", f"{execution_time: .4f}s")

    start_time = time.perf_counter()
    result = bottomup(rgb, energy, ratio)
    end_time = time.perf_counter()

    execution_time = end_time - start_time

    print("bottomup time:", f"{execution_time: .4f}s")

    return result


img = imread("../data/input/turtle/rgb.jpg")
img = cvtColor(img, COLOR_BGR2RGB)

gray = cvtColor(img, COLOR_BGR2GRAY)
gray = sobel(gray)

gt = imread("../data/input/turtle/gt.jpg")
gt = cvtColor(gt, COLOR_BGR2GRAY)

energy_map = np.maximum(gray, gt)

# Plotter.images([gray, gt, energy_map], 1, 3)

r = 0.75

image = SeamCarving(img, energy_map, r)

Plotter.image(image)
