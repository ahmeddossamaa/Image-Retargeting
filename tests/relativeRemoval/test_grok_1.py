# 05:28 AM - 11/6/2025

def remove_elements(arr, k):
    """
    Remove k elements from array to minimize shift of significant elements.
    Returns the resulting array with significant elements in relative positions.

    Args:
        arr (list): Input array
        k (int): Number of elements to remove

    Returns:
        list: Array with k elements removed
    """
    if k >= len(arr):
        return []

    if k < 0:
        raise ValueError("k must be non-negative")

    # Find the most frequent element (assumed to be filler, e.g., 0 or 1)
    from collections import Counter
    counts = Counter(arr)
    filler = counts.most_common(1)[0][0]  # Most frequent element

    # Identify significant elements (non-filler)
    significant_indices = [i for i, x in enumerate(arr) if x != filler]

    # If no significant elements, remove k elements from ends
    if not significant_indices:
        left_remove = k // 2
        right_remove = k - left_remove
        return arr[left_remove:len(arr) - right_remove]

    # Count filler elements on left and right
    left_filler = 0
    for i in range(significant_indices[0]):
        if arr[i] == filler:
            left_filler += 1

    right_filler = 0
    for i in range(significant_indices[-1] + 1, len(arr)):
        if arr[i] == filler:
            right_filler += 1

    # Distribute k removals
    total_filler = left_filler + right_filler
    if total_filler < k:
        raise ValueError("Not enough filler elements to remove")

    # Try to balance removals
    left_remove = min(left_filler, k // 2)
    right_remove = min(right_filler, k - left_remove)

    # Adjust if we haven't removed enough
    if left_remove + right_remove < k:
        additional = k - (left_remove + right_remove)
        if left_filler > left_remove:
            extra_left = min(left_filler - left_remove, additional)
            left_remove += extra_left
            additional -= extra_left
        if additional > 0 and right_filler > right_remove:
            right_remove += min(right_filler - right_remove, additional)

    # Construct the result
    start = left_remove
    end = len(arr) - right_remove
    return arr[start:end]


# Test cases
if __name__ == "__main__":
    test_cases = [
        ([0, 0, 0, 12, 0, 0, 0], 4),
        ([1, 1, 1, 8, 1, 1, 1, 1, 1, 1], 6),
        ([0, 1, 0, 2, 9, 1, 1, 0, 2, 12, 0, 0], 5)
    ]

    for arr, k in test_cases:
        result = remove_elements(arr, k)
        print(f"Input: {arr}, k={k}, Output: {result}")