from typing import List, Optional, TypeVar, Union

T = TypeVar('T')


def counting_sort(arr: List[int], max_val: Optional[int] = None) -> List[int]:
    """
    Counting sort algorithm for non-negative integers.

    Args:
        arr: List of non-negative integers to sort
        max_val: Maximum value in the array (calculated if not provided)

    Returns:
        Sorted list
    """
    if not arr:
        return []

    if max_val is None:
        max_val = max(arr)

    # Create count array
    count = [0] * (max_val + 1)

    # Count occurrences
    for num in arr:
        count[num] += 1

    # Reconstruct the sorted array
    sorted_arr = []
    for i in range(max_val + 1):
        sorted_arr.extend([i] * count[i])

    return sorted_arr


def radix_sort(arr: List[int]) -> List[int]:
    """
    Radix sort algorithm for non-negative integers.

    Args:
        arr: List of non-negative integers to sort

    Returns:
        Sorted list
    """
    if not arr:
        return []

    # Find the maximum number to know the number of digits
    max_val = max(arr)

    # Do counting sort for every digit
    exp = 1
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr


def counting_sort_by_digit(arr: List[int], exp: int) -> List[int]:
    """Helper function for radix sort."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    # Count occurrences of each digit
    for num in arr:
        index = (num // exp) % 10
        count[index] += 1

    # Change count[i] to contain actual position of this digit in output
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    return output


def bucket_sort(arr: List[Union[int, float]], bucket_size: int = 5) -> List[Union[int, float]]:
    """
    Bucket sort algorithm for sorting values between 0 and 1.

    Args:
        arr: List of values to sort
        bucket_size: Number of buckets

    Returns:
        Sorted list
    """
    if not arr:
        return []

    # Find minimum and maximum values
    min_val, max_val = min(arr), max(arr)

    # Create empty buckets
    bucket_count = (max_val - min_val) // bucket_size + 1
    buckets = [[] for _ in range(int(bucket_count))]

    # Put elements in buckets
    for num in arr:
        index = int((num - min_val) // bucket_size)
        buckets[index].append(num)

    # Sort each bucket and concatenate
    result = []
    for bucket in buckets:
        # Use insertion sort for each bucket
        insertion_sort(bucket)
        result.extend(bucket)

    return result


def insertion_sort(arr: List[T]) -> None:
    """
    In-place insertion sort algorithm (helper for bucket sort).

    Args:
        arr: List to sort
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def pigeonhole_sort(arr: List[int]) -> List[int]:
    """
    Pigeonhole sort algorithm for integers.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list
    """
    if not arr:
        return []

    # Find minimum and maximum values
    min_val, max_val = min(arr), max(arr)
    size = max_val - min_val + 1

    # Create pigeonholes
    holes = [0] * size

    # Fill the pigeonholes
    for num in arr:
        holes[num - min_val] += 1

    # Put elements back into the array
    result = []
    for i in range(size):
        result.extend([i + min_val] * holes[i])

    return result


def counting_sort_for_strings(strings: List[str], pos: int) -> List[str]:
    """
    Counting sort algorithm for sorting strings by a specific position.

    Args:
        strings: List of strings to sort
        pos: Position (index) to sort by

    Returns:
        Sorted list of strings
    """
    n = len(strings)

    # Count array for all ASCII characters
    count = [0] * 256

    # Count occurrences of each character at position pos
    for s in strings:
        if pos < len(s):
            count[ord(s[pos])] += 1
        else:
            # If the string is shorter than pos, use 0 (lowest value)
            count[0] += 1

    # Change count to cumulative count
    for i in range(1, 256):
        count[i] += count[i - 1]

    # Build the output array
    output = [""] * n
    for i in range(n - 1, -1, -1):
        if pos < len(strings[i]):
            idx = ord(strings[i][pos])
        else:
            idx = 0
        output[count[idx] - 1] = strings[i]
        count[idx] -= 1

    return output


def radix_sort_for_strings(strings: List[str]) -> List[str]:
    """
    Radix sort algorithm for strings.

    Args:
        strings: List of strings to sort

    Returns:
        Sorted list of strings
    """
    if not strings:
        return []

    # Find the maximum length string
    max_len = max(len(s) for s in strings)

    # Sort strings using counting sort from least significant character to most
    result = strings
    for pos in range(max_len - 1, -1, -1):
        result = counting_sort_for_strings(result, pos)

    return result