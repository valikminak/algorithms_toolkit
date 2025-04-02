from typing import List, TypeVar

T = TypeVar('T')


def quick_sort(arr: List[T]) -> List[T]:
    """
    Quick sort algorithm implementation.

    Args:
        arr: List to be sorted

    Returns:
        Sorted list
    """
    if len(arr) <= 1:
        return arr.copy()

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: List[T]) -> List[T]:
    """
    Merge sort algorithm implementation.

    Args:
        arr: List to be sorted

    Returns:
        Sorted list
    """
    if len(arr) <= 1:
        return arr.copy()

    # Divide the array into two halves
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    # Recursively sort both halves
    left = merge_sort(left)
    right = merge_sort(right)

    # Merge the sorted halves
    return merge(left, right)


def merge(left: List[T], right: List[T]) -> List[T]:
    """
    Merge two sorted arrays.

    Args:
        left: First sorted array
        right: Second sorted array

    Returns:
        Merged sorted array
    """
    result = []
    i = j = 0

    # Compare elements from both arrays and add the smaller one to the result
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])

    return result


def heap_sort(arr: List[T]) -> List[T]:
    """
    Heap sort algorithm implementation.

    Args:
        arr: List to be sorted

    Returns:
        Sorted list
    """
    arr = arr.copy()  # Make a copy to avoid modifying the input
    n = len(arr)

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Swap
        heapify(arr, i, 0)

    return arr


def heapify(arr: List[T], n: int, i: int) -> None:
    """
    Heapify a subtree rooted at index i.

    Args:
        arr: Array to heapify
        n: Size of the heap
        i: Index of the root of the subtree
    """
    largest = i  # Initialize largest as root
    left = 2 * i + 1
    right = 2 * i + 2

    # Check if left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left

    # Check if right child exists and is greater than root
    if right < n and arr[right] > arr[largest]:
        largest = right

    # Change root if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Swap
        heapify(arr, n, largest)


def insertion_sort(arr: List[T]) -> List[T]:
    """
    Insertion sort algorithm implementation.

    Args:
        arr: List to be sorted

    Returns:
        Sorted list
    """
    arr = arr.copy()  # Make a copy to avoid modifying the input

    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr


def selection_sort(arr: List[T]) -> List[T]:
    """
    Selection sort algorithm implementation.

    Args:
        arr: List to be sorted

    Returns:
        Sorted list
    """
    arr = arr.copy()  # Make a copy to avoid modifying the input
    n = len(arr)

    for i in range(n):
        # Find the minimum element in the unsorted part
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # Swap the found minimum element with the first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr


def bubble_sort(arr: List[T]) -> List[T]:
    """
    Bubble sort algorithm implementation.

    Args:
        arr: List to be sorted

    Returns:
        Sorted list
    """
    arr = arr.copy()  # Make a copy to avoid modifying the input
    n = len(arr)

    for i in range(n):
        # Flag to optimize by stopping early if no swaps occur
        swapped = False

        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no swaps occurred in this pass, the array is sorted
        if not swapped:
            break

    return arr


def shell_sort(arr: List[T]) -> List[T]:
    """
    Shell sort algorithm implementation.

    Args:
        arr: List to be sorted

    Returns:
        Sorted list
    """
    arr = arr.copy()  # Make a copy to avoid modifying the input
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i

            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap

            arr[j] = temp

        gap //= 2

    return arr


def tim_sort(arr: List[T]) -> List[T]:
    """
    Tim sort algorithm implementation.

    Tim sort is a hybrid sorting algorithm derived from merge sort and insertion sort.
    In Python, the built-in sorted() function and list.sort() method use Tim sort.

    Args:
        arr: List to be sorted

    Returns:
        Sorted list
    """
    # For simplicity, we'll use Python's built-in sort which is Tim sort
    arr_copy = arr.copy()
    arr_copy.sort()
    return arr_copy


def intro_sort(arr: List[T]) -> List[T]:
    """
    Intro sort algorithm implementation.

    Intro sort is a hybrid sorting algorithm that uses quick sort, heap sort,
    and insertion sort to achieve both fast average performance and optimal worst-case performance.

    Args:
        arr: List to be sorted

    Returns:
        Sorted list
    """
    arr = arr.copy()  # Make a copy to avoid modifying the input

    # Calculate max recursion depth
    max_depth = 2 * (len(arr).bit_length())

    # Call the helper function
    _intro_sort(arr, 0, len(arr) - 1, max_depth)

    return arr


def _intro_sort(arr: List[T], start: int, end: int, max_depth: int) -> None:
    """Helper function for intro sort."""
    # If the array size is small, use insertion sort
    if end - start + 1 < 16:
        _insertion_sort_range(arr, start, end)
        return

    # If max recursion depth reached, use heap sort
    if max_depth == 0:
        _heap_sort_range(arr, start, end)
        return

    # Otherwise, use quick sort
    pivot = _partition(arr, start, end)
    _intro_sort(arr, start, pivot - 1, max_depth - 1)
    _intro_sort(arr, pivot + 1, end, max_depth - 1)


def _partition(arr: List[T], start: int, end: int) -> int:
    """Partition function for quick sort."""
    # Choose pivot (median of three)
    mid = (start + end) // 2
    pivot = median_of_three(arr[start], arr[mid], arr[end])

    # Find pivot index
    pivot_idx = start
    if arr[mid] == pivot:
        pivot_idx = mid
    elif arr[end] == pivot:
        pivot_idx = end

    # Swap pivot to the end
    arr[pivot_idx], arr[end] = arr[end], arr[pivot_idx]

    # Partition the array
    i = start - 1
    for j in range(start, end):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # Swap pivot back
    arr[i + 1], arr[end] = arr[end], arr[i + 1]

    return i + 1


def median_of_three(a: T, b: T, c: T) -> T:
    """Find the median of three values."""
    if (a <= b <= c) or (c <= b <= a):
        return b
    elif (b <= a <= c) or (c <= a <= b):
        return a
    else:
        return c


def _insertion_sort_range(arr: List[T], start: int, end: int) -> None:
    """Insertion sort for a range of an array."""
    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        while j >= start and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def _heap_sort_range(arr: List[T], start: int, end: int) -> None:
    """Heap sort for a range of an array."""
    # Build heap
    for i in range(start + (end - start) // 2, start - 1, -1):
        _heapify(arr, i, end, start)

    # Extract elements
    for i in range(end, start, -1):
        arr[start], arr[i] = arr[i], arr[start]
        _heapify(arr, start, i - 1, start)


def _heapify(arr: List[T], root: int, end: int, start: int) -> None:
    """Heapify a specific range of an array."""
    largest = root
    left = 2 * (root - start) + 1 + start
    right = 2 * (root - start) + 2 + start

    if left <= end and arr[left] > arr[largest]:
        largest = left

    if right <= end and arr[right] > arr[largest]:
        largest = right

    if largest != root:
        arr[root], arr[largest] = arr[largest], arr[root]
        _heapify(arr, largest, end, start)
