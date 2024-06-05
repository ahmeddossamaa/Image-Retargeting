import numpy as np


def shift_half_left_1d(array):
    n = len(array)
    mid = 0

    # Split the array into two halves
    first_half = array[:mid]
    second_half = array[mid:]

    print(first_half)
    print(second_half)

    # Shift the first half left by one position
    # shifted_first_half = np.roll(first_half, -1)
    shifted_second_half = np.roll(second_half, -1)

    # Combine the shifted first half with the unchanged second half
    # shifted_array = np.concatenate((shifted_first_half, second_half))
    shifted_array = np.concatenate((first_half, shifted_second_half))
    return shifted_array


# Example usage
arr = np.array([np.inf, 2, 3, 4, 5, 6, 7, 8])
shifted_arr = shift_half_left_1d(arr)
print("Original array:", arr)
print("Shifted array:", shifted_arr)
