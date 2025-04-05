# searching/binary.py
from typing import List, Optional, TypeVar
import math

T = TypeVar('T')


def binary_search(arr: List[T], target: T) -> int:
    """
    Binary search algorithm for sorted arrays.

    Args:
        arr: Sorted list of elements
        target: The value to search for

    Returns:
        Index of the target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def binary_search_recursive(arr: List[T], target: T, left: int = 0, right: Optional[int] = None) -> int:
    """
    Recursive binary search algorithm for sorted arrays.

    Args:
        arr: Sorted list of elements
        target: The value to search for
        left: Left boundary index
        right: Right boundary index

    Returns:
        Index of the target if found, -1 otherwise
    """
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


def lower_bound(arr: List[T], target: T) -> int:
    """
    Find the first position where target could be inserted without changing the order.

    Args:
        arr: Sorted list of elements
        target: The value to search for

    Returns:
        Index of the first element not less than target
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left


def upper_bound(arr: List[T], target: T) -> int:
    """
    Find the first position where target+1 could be inserted without changing the order.

    Args:
        arr: Sorted list of elements
        target: The value to search for

    Returns:
        Index of the first element greater than target
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid

    return left


def binary_search_first_occurrence(arr: List[T], target: T) -> int:
    """
    Find the first occurrence of target in a sorted array with duplicates.

    Args:
        arr: Sorted list of elements
        target: The value to search for

    Returns:
        Index of the first occurrence of target, -1 if not found
    """
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid  # Save the result
            right = mid - 1  # Continue searching on the left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result


def binary_search_last_occurrence(arr: List[T], target: T) -> int:
    """
    Find the last occurrence of target in a sorted array with duplicates.

    Args:
        arr: Sorted list of elements
        target: The value to search for

    Returns:
        Index of the last occurrence of target, -1 if not found
    """
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid  # Save the result
            left = mid + 1  # Continue searching on the right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result


def exponential_search(arr: List[T], target: T) -> int:
    """
    Exponential search algorithm for sorted arrays.

    It's particularly useful for unbounded arrays or arrays where the target
    is more likely at the beginning.

    Args:
        arr: Sorted list of elements
        target: The value to search for

    Returns:
        Index of the target if found, -1 otherwise
    """
    n = len(arr)

    # If array is empty
    if n == 0:
        return -1

    # If target is at first position
    if arr[0] == target:
        return 0

    # Find range for binary search by repeated doubling
    i = 1
    while i < n and arr[i] <= target:
        i *= 2

    # Call binary search for the found range
    return binary_search(arr, target, i // 2, min(i, n - 1))


def jump_search(arr: List[T], target: T) -> int:
    """
    Jump search algorithm for sorted arrays.

    It works by jumping ahead by fixed steps and then performing a linear search.

    Args:
        arr: Sorted list of elements
        target: The value to search for

    Returns:
        Index of the target if found, -1 otherwise
    """
    n = len(arr)

    # Finding block size to be jumped
    step = int(math.sqrt(n))

    # Finding the block where element is present (if it is present)
    prev = 0
    while prev < n and arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1

    # Doing a linear search for target in block beginning with prev
    while prev < min(step, n) and arr[prev] < target:
        prev += 1

    # If element is found
    if prev < n and arr[prev] == target:
        return prev

    return -1


def interpolation_search(arr: List[int], target: int) -> int:
    """
    Interpolation search algorithm for uniformly distributed sorted arrays.

    It uses a formula to estimate the position of the target value.

    Args:
        arr: Sorted list of integers
        target: The value to search for

    Returns:
        Index of the target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1

    while left <= right and arr[left] <= target <= arr[right]:
        if arr[left] == arr[right]:  # If all elements in the range are same
            if arr[left] == target:
                return left
            return -1

        # Formula for interpolation search
        pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])

        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1

    return -1


def fibonacci_search(arr: List[T], target: T) -> int:
    """
    Fibonacci search algorithm for sorted arrays.

    It uses Fibonacci numbers to divide the array.

    Args:
        arr: Sorted list of elements
        target: The value to search for

    Returns:
        Index of the target if found, -1 otherwise
    """
    n = len(arr)

    # Initialize Fibonacci numbers
    fib_m2 = 0  # (m-2)'th Fibonacci number
    fib_m1 = 1  # (m-1)'th Fibonacci number
    fib = fib_m1 + fib_m2  # m'th Fibonacci number

    # Find the smallest Fibonacci number greater than or equal to n
    while fib < n:
        fib_m2 = fib_m1
        fib_m1 = fib
        fib = fib_m1 + fib_m2

    # Marks the eliminated range from front
    offset = -1

    # While there are elements to be inspected
    while fib > 1:
        # Check if fib_m2 is a valid index
        i = min(offset + fib_m2, n - 1)

        # If target is greater than the value at index i, cut the array from offset to i
        if arr[i] < target:
            fib = fib_m1
            fib_m1 = fib_m2
            fib_m2 = fib - fib_m1
            offset = i
        # If target is less than the value at index i, cut the array after i+1
        elif arr[i] > target:
            fib = fib_m2
            fib_m1 = fib_m1 - fib_m2
            fib_m2 = fib - fib_m1
        # Element found
        else:
            return i

    # Compare the last element
    if fib_m1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1

    return -1