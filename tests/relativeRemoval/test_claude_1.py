# 05:30 AM - 11/6/2025

def remove_elements_preserve_position(arr, k, important_value_threshold=0):
    """
    Remove k elements from array while preserving relative positions of important values.

    Args:
        arr: Input array
        k: Number of elements to remove
        important_value_threshold: Values above this threshold are considered important

    Returns:
        Modified array with k elements removed
    """
    if k >= len(arr):
        return []

    if k == 0:
        return arr.copy()

    # Identify important elements and their positions
    important_positions = []
    for i, val in enumerate(arr):
        if abs(val) > important_value_threshold:
            important_positions.append(i)

    if not important_positions:
        # No important elements, remove from start
        return arr[k:]

    # Calculate target positions for important elements in the result array
    result_length = len(arr) - k
    target_positions = []

    if len(important_positions) == 1:
        # Single important element - try to keep it centered
        original_pos = important_positions[0]
        original_length = len(arr)
        # Maintain relative position as closely as possible
        target_pos = int(original_pos * result_length / original_length)
        target_positions.append(target_pos)
    else:
        # Multiple important elements - maintain relative spacing
        first_important = important_positions[0]
        last_important = important_positions[-1]
        original_span = last_important - first_important

        if original_span == 0:
            # All important elements at same position
            target_positions = [important_positions[0] * result_length // len(arr)] * len(important_positions)
        else:
            # Scale the span to fit in result array
            # Leave some padding on both sides if possible
            available_span = result_length - 1
            if available_span <= 0:
                available_span = result_length - 1 if result_length > 1 else 0

            # Try to center the important elements
            if len(important_positions) > 1:
                # Calculate scaling factor
                scale_factor = min(1.0, available_span / original_span) if original_span > 0 else 1.0

                # Calculate offset to center the group
                scaled_span = original_span * scale_factor
                offset = max(0, (result_length - 1 - scaled_span) / 2)

                for pos in important_positions:
                    relative_pos = pos - first_important
                    target_pos = int(offset + relative_pos * scale_factor)
                    target_pos = max(0, min(result_length - 1, target_pos))
                    target_positions.append(target_pos)

    # Create result array
    result = [0] * result_length  # Initialize with default values

    # Place important elements at target positions
    important_elements = [arr[pos] for pos in important_positions]
    for i, (target_pos, element) in enumerate(zip(target_positions, important_elements)):
        if 0 <= target_pos < result_length:
            result[target_pos] = element

    # Fill remaining positions with non-important elements
    non_important_elements = []
    for i, val in enumerate(arr):
        if i not in important_positions:
            non_important_elements.append(val)

    # Fill empty positions in result
    non_important_idx = 0
    for i in range(result_length):
        if i not in target_positions and non_important_idx < len(non_important_elements):
            result[i] = non_important_elements[non_important_idx]
            non_important_idx += 1

    return result


def remove_elements_balanced(arr, k):
    """
    Alternative approach: Remove elements by balancing from both ends.
    """
    if k >= len(arr):
        return []

    if k == 0:
        return arr.copy()

    # Remove elements from both ends as evenly as possible
    left_remove = k // 2
    right_remove = k - left_remove

    start_idx = left_remove
    end_idx = len(arr) - right_remove

    return arr[start_idx:end_idx]


# Test the functions with your examples
def test_examples():
    print("Testing with your examples:")
    print("=" * 50)

    # Example 1
    arr1 = [0, 0, 0, 12, 0, 0, 0]
    k1 = 4
    result1 = remove_elements_preserve_position(arr1, k1)
    balanced1 = remove_elements_balanced(arr1, k1)
    print(f"Example 1: {arr1}")
    print(f"Remove {k1} elements")
    print(f"Position-preserving result: {result1}")
    print(f"Balanced removal result: {balanced1}")
    print(f"Expected: [0, 12, 0]")
    print()

    # Example 2
    arr2 = [1, 1, 1, 8, 1, 1, 1, 1, 1, 1]
    k2 = 6
    result2 = remove_elements_preserve_position(arr2, k2, important_value_threshold=1)
    balanced2 = remove_elements_balanced(arr2, k2)
    print(f"Example 2: {arr2}")
    print(f"Remove {k2} elements")
    print(f"Position-preserving result: {result2}")
    print(f"Balanced removal result: {balanced2}")
    print(f"Expected: [1, 8, 1, 1]")
    print()

    # Example 3
    # arr3 = [0, 1, 0, 2, 9, 1, 1, 0, 2, 12, 0, 0]
    arr3 = [0, 12, 0, 1, 1, 11, 5, 2, 6, 0, 0, 1, 3, 7, 1]
    k3 = 7
    result3 = remove_elements_preserve_position(arr3, k3)
    balanced3 = remove_elements_balanced(arr3, k3)
    print(f"Example 3: {arr3}")
    print(f"Remove {k3} elements")
    print(f"Position-preserving result: {result3}")
    print(f"Balanced removal result: {balanced3}")
    print(f"Expected: [1, 2, 9, 1, 2, 12, 0]")


# Run the tests
if __name__ == "__main__":
    test_examples()

# Mathematical equation for position calculation:
"""
For maintaining relative positions of important elements:

Given:
- Original array length: n
- Target array length: m = n - k (where k is elements to remove)
- Important element at original position: p_orig
- Important elements span from position: p_first to p_last

Position calculation:
1. For single important element:
   p_target = floor(p_orig * m / n)

2. For multiple important elements:
   scale_factor = min(1.0, (m-1) / (p_last - p_first))
   offset = (m - 1 - (p_last - p_first) * scale_factor) / 2
   p_target = floor(offset + (p_orig - p_first) * scale_factor)

Where:
- p_target is constrained to [0, m-1]
- scale_factor maintains relative spacing
- offset centers the important elements in the result array
"""