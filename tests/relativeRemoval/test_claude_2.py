# 05:44 AM - 11/6/2025

import numpy as np
from statistics import mean, median, stdev
from collections import Counter


def identify_important_elements(arr, strategy='statistical'):
    """
    Identify important elements using various strategies.

    Args:
        arr: Input array
        strategy: 'statistical', 'outliers', 'top_values', 'adaptive'

    Returns:
        List of indices of important elements
    """
    if not arr:
        return []

    arr_abs = [abs(x) for x in arr]

    if strategy == 'statistical':
        # Use mean + standard deviation threshold
        if len(set(arr_abs)) == 1:  # All values are the same
            return []

        avg = mean(arr_abs)
        try:
            std = stdev(arr_abs) if len(arr_abs) > 1 else 0
        except:
            std = 0

        threshold = avg + std * 0.5  # Adjust multiplier as needed
        important_indices = [i for i, val in enumerate(arr) if abs(val) > threshold]
        return important_indices

    elif strategy == 'outliers':
        # Use IQR method to detect outliers
        sorted_vals = sorted(arr_abs)
        n = len(sorted_vals)

        if n < 4:
            # Not enough data for IQR, use top values
            max_val = max(arr_abs)
            return [i for i, val in enumerate(arr) if abs(val) == max_val]

        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1

        if iqr == 0:
            # No variation, keep maximum values
            max_val = max(arr_abs)
            return [i for i, val in enumerate(arr) if abs(val) == max_val]

        threshold = q3 + 1.5 * iqr
        important_indices = [i for i, val in enumerate(arr) if abs(val) > threshold]
        return important_indices

    elif strategy == 'top_values':
        # Keep top N% of values by magnitude
        unique_vals = sorted(set(arr_abs), reverse=True)
        if len(unique_vals) <= 2:
            # Keep the maximum values
            max_val = max(arr_abs)
            return [i for i, val in enumerate(arr) if abs(val) == max_val]

        # Keep roughly top 20-30% of unique values
        top_count = max(1, len(unique_vals) // 3)
        top_values = set(unique_vals[:top_count])
        important_indices = [i for i, val in enumerate(arr) if abs(val) in top_values]
        return important_indices

    elif strategy == 'adaptive':
        # Combine multiple strategies
        statistical_indices = set(identify_important_elements(arr, 'statistical'))
        outlier_indices = set(identify_important_elements(arr, 'outliers'))
        top_indices = set(identify_important_elements(arr, 'top_values'))

        # Use union of all methods, but prioritize outliers and top values
        important_indices = list(outlier_indices | top_indices | statistical_indices)

        # If too many elements are marked as important, prioritize by value
        if len(important_indices) > len(arr) * 0.4:  # More than 40% marked as important
            # Keep only the truly outstanding values
            values_at_indices = [(abs(arr[i]), i) for i in important_indices]
            values_at_indices.sort(reverse=True)
            # Keep top 25% or at least 1 element
            keep_count = max(1, len(important_indices) // 4)
            important_indices = [idx for _, idx in values_at_indices[:keep_count]]

        return important_indices

    return []


def remove_elements_preserve_position(arr, k, strategy='adaptive'):
    """
    Remove k elements from array while preserving positions of important values.

    Args:
        arr: Input array
        k: Number of elements to remove
        strategy: Strategy for identifying important elements

    Returns:
        Modified array with k elements removed
    """
    if k >= len(arr):
        return []

    if k == 0:
        return arr.copy()

    # Identify important elements using the specified strategy
    important_positions = identify_important_elements(arr, strategy)

    if not important_positions:
        # No important elements identified, remove from start
        return arr[k:]

    result_length = len(arr) - k

    # Ensure we don't have more important elements than result length
    if len(important_positions) > result_length:
        # Prioritize by absolute value
        values_at_positions = [(abs(arr[pos]), pos) for pos in important_positions]
        values_at_positions.sort(reverse=True)
        important_positions = [pos for _, pos in values_at_positions[:result_length]]

    # Calculate target positions for important elements
    target_positions = calculate_target_positions(important_positions, len(arr), result_length)

    # Build result array
    result = [0] * result_length
    used_positions = set()

    # Place important elements at target positions
    for orig_pos, target_pos in zip(important_positions, target_positions):
        if 0 <= target_pos < result_length and target_pos not in used_positions:
            result[target_pos] = arr[orig_pos]
            used_positions.add(target_pos)

    # Fill remaining positions with non-important elements
    non_important_elements = []
    for i, val in enumerate(arr):
        if i not in important_positions:
            non_important_elements.append(val)

    non_important_idx = 0
    for i in range(result_length):
        if i not in used_positions and non_important_idx < len(non_important_elements):
            result[i] = non_important_elements[non_important_idx]
            non_important_idx += 1

    return result


def calculate_target_positions(important_positions, original_length, result_length):
    """Calculate target positions for important elements."""
    if not important_positions:
        return []

    if len(important_positions) == 1:
        # Single element - maintain relative position
        original_pos = important_positions[0]
        target_pos = int(original_pos * result_length / original_length)
        return [min(result_length - 1, target_pos)]

    # Multiple elements - maintain relative spacing
    important_positions.sort()
    first_pos = important_positions[0]
    last_pos = important_positions[-1]
    original_span = last_pos - first_pos

    if original_span == 0:
        # All at same position
        target = int(first_pos * result_length / original_length)
        return [min(result_length - 1, target)] * len(important_positions)

    # Scale and center the elements
    available_span = result_length - 1
    scale_factor = min(1.0, available_span / original_span) if original_span > 0 else 1.0
    scaled_span = original_span * scale_factor
    offset = max(0, (available_span - scaled_span) / 2)

    target_positions = []
    for pos in important_positions:
        relative_pos = pos - first_pos
        target_pos = int(offset + relative_pos * scale_factor)
        target_pos = max(0, min(result_length - 1, target_pos))
        target_positions.append(target_pos)

    return target_positions


def analyze_array_statistics(arr):
    """Analyze array to show why certain elements are considered important."""
    if not arr:
        return

    arr_abs = [abs(x) for x in arr]
    print(f"Array: {arr}")
    print(f"Absolute values: {arr_abs}")
    print(f"Mean: {mean(arr_abs):.2f}")
    print(f"Median: {median(arr_abs):.2f}")
    if len(arr_abs) > 1:
        print(f"Std Dev: {stdev(arr_abs):.2f}")
    print(f"Max: {max(arr_abs)}")
    print(f"Min: {min(arr_abs)}")

    print("\nImportant elements by strategy:")
    for strategy in ['statistical', 'outliers', 'top_values', 'adaptive']:
        important_indices = identify_important_elements(arr, strategy)
        important_values = [arr[i] for i in important_indices]
        print(f"  {strategy}: indices {important_indices} -> values {important_values}")
    print()


# Test function with detailed analysis
def test_examples_with_analysis():
    print("Array Element Removal with Statistical Analysis")
    print("=" * 60)

    test_cases = [
        ([0, 0, 0, 12, 0, 0, 0], 4, [0, 12, 0]),
        ([1, 1, 1, 8, 1, 1, 1, 1, 1, 1], 6, [1, 8, 1, 1]),
        ([0, 1, 0, 2, 9, 1, 1, 0, 2, 12, 0, 0], 5, [1, 2, 9, 1, 2, 12, 0]),
        # Additional test case with large values
        ([1, 2, 1, 50, 1, 100, 1, 2], 3, None),  # Should preserve 50 and 100
        ([5, 5, 5, 30, 5, 5, 200, 5], 4, None),  # Should preserve 30 and 200
    ]

    for i, (arr, k, expected) in enumerate(test_cases, 1):
        print(f"\n=== Example {i} ===")
        analyze_array_statistics(arr)

        print(f"Removing {k} elements:")

        for strategy in ['adaptive', 'statistical', 'outliers', 'top_values']:
            result = remove_elements_preserve_position(arr, k, strategy)
            print(f"  {strategy:12}: {result}")

        if expected:
            print(f"  {'Expected':12}: {expected}")
        print()


# Quick test function
def quick_test(arr, k, strategy='adaptive'):
    """Quick test function for interactive use."""
    print(f"Original: {arr}")
    print(f"Remove {k} elements using '{strategy}' strategy:")

    important_indices = identify_important_elements(arr, strategy)
    important_values = [arr[i] for i in important_indices]
    print(f"Important elements: indices {important_indices} -> values {important_values}")

    result = remove_elements_preserve_position(arr, k, strategy)
    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    test_examples_with_analysis()

    # Interactive testing
    print("\n" + "=" * 60)
    print("Interactive Testing:")
    print("Use quick_test([your_array], k, 'strategy') to test specific cases")
    print("Available strategies: 'statistical', 'outliers', 'top_values', 'adaptive'")

    # Example of problematic case you mentioned
    print("\nTesting problematic case where large values might be removed:")
    test_array = [1, 2, 3, 100, 2, 1, 50, 1]
    quick_test(test_array, 3, 'adaptive')