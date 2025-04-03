from flask import Blueprint, jsonify, request
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import your algorithms
from kosmos.sorting import bubble_sort, quick_sort, merge_sort
from utils.performance import benchmark, measure_execution_time
from utils.matplotlib_adapter import convert_plot_to_image

sorting_bp = Blueprint('sorting', __name__)


@sorting_bp.route('/algorithms')
def get_sorting_algorithms():
    """Return available sorting algorithms"""
    algorithms = [
        {"id": "bubble_sort", "name": "Bubble Sort", "complexity": "O(n²)"},
        {"id": "quick_sort", "name": "Quick Sort", "complexity": "O(n log n)"},
        {"id": "merge_sort", "name": "Merge Sort", "complexity": "O(n log n)"},
        # Add more algorithms as available
    ]
    return jsonify(algorithms)


@sorting_bp.route('/run', methods=['POST'])
def run_sorting():
    """Run a sorting algorithm and return the results"""
    data = request.json
    algorithm_name = data.get('algorithm', 'bubble_sort')
    input_array = data.get('input', [5, 3, 8, 1, 2, 9])

    # Map algorithm names to functions
    algorithms = {
        'bubble_sort': bubble_sort,
        'quick_sort': quick_sort,
        'merge_sort': merge_sort,
    }

    algorithm = algorithms.get(algorithm_name)
    if not algorithm:
        return jsonify({'error': 'Algorithm not found'}), 404

    # Measure execution time
    result, execution_time = measure_execution_time(algorithm, input_array.copy())

    # Generate visualization frames
    frames = generate_sorting_frames(algorithm_name, input_array.copy())

    return jsonify({
        'algorithm': algorithm_name,
        'input': input_array,
        'output': result,
        'execution_time': execution_time,
        'visualization': frames,
        'category': 'sorting'
    })


@sorting_bp.route('/compare', methods=['POST'])
def compare_sorting():
    """Compare multiple sorting algorithms"""
    data = request.json
    algorithm_names = data.get('algorithms', ['bubble_sort', 'quick_sort', 'merge_sort'])
    input_array = data.get('input', [5, 3, 8, 1, 2, 9])

    algorithms = {
        'bubble_sort': bubble_sort,
        'quick_sort': quick_sort,
        'merge_sort': merge_sort,
    }

    # Filter out any invalid algorithm names
    selected_algorithms = [algorithms[name] for name in algorithm_names if name in algorithms]

    if not selected_algorithms:
        return jsonify({'error': 'No valid algorithms selected'}), 400

    # Run benchmark
    results = benchmark(
        selected_algorithms,
        [input_array.copy()],
        labels=[name for name in algorithm_names if name in algorithms]
    )

    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [times[0] for times in results.values()])
    plt.title('Algorithm Performance Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()

    # Convert plot to image
    img_data = convert_plot_to_image(plt)
    plt.close()

    return jsonify({
        'algorithms': algorithm_names,
        'execution_times': {k: v[0] for k, v in results.items()},
        'comparison_chart': img_data
    })


@sorting_bp.route('/code', methods=['GET'])
def get_algorithm_code():
    """Return the code implementation for a sorting algorithm"""
    algorithm_name = request.args.get('algorithm', 'bubble_sort')

    code_samples = {
        'bubble_sort': '''
def bubble_sort(arr):
    """
    Bubble sort implementation.

    Args:
        arr: The array to sort

    Returns:
        The sorted array
    """
    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no swapping occurred in this pass, array is sorted
        if not swapped:
            break

    return arr
''',
        'quick_sort': '''
def quick_sort(arr):
    """
    Quick sort implementation.

    Args:
        arr: The array to sort

    Returns:
        The sorted array
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)
''',
        'merge_sort': '''
def merge_sort(arr):
    """
    Merge sort implementation.

    Args:
        arr: The array to sort

    Returns:
        The sorted array
    """
    if len(arr) <= 1:
        return arr

    # Split array in half
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    # Recursively sort both halves
    left = merge_sort(left)
    right = merge_sort(right)

    # Merge the sorted halves
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays"""
    result = []
    i = j = 0

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
'''
    }

    return jsonify({
        'code': code_samples.get(algorithm_name, 'Code not available for this algorithm')
    })


def generate_sorting_frames(algorithm_name, input_array):
    """Generate visualization frames for sorting algorithms directly"""

    if algorithm_name == 'bubble_sort':
        return generate_bubble_sort_frames(input_array)
    elif algorithm_name == 'quick_sort':
        frames = []
        quick_sort_with_frames(input_array, 0, len(input_array) - 1, frames)
        return frames
    elif algorithm_name == 'merge_sort':
        frames = []
        merge_sort_with_frames(input_array, frames)
        return frames
    else:
        # Fallback - just show before and after
        return [
            {'state': input_array, 'info': 'Initial array', 'highlight': []},
            {'state': sorted(input_array), 'info': 'Sorted array', 'highlight': []}
        ]


def generate_bubble_sort_frames(arr):
    """Generate frames for visualizing bubble sort"""
    frames = []
    array = arr.copy()
    n = len(array)

    # Initial state
    frames.append({
        'state': array.copy(),
        'info': 'Initial array',
        'highlight': []
    })

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            # Comparing elements
            frames.append({
                'state': array.copy(),
                'info': f'Comparing {array[j]} and {array[j + 1]}',
                'highlight': [j, j + 1]
            })

            if array[j] > array[j + 1]:
                # Swap elements
                array[j], array[j + 1] = array[j + 1], array[j]
                swapped = True

                frames.append({
                    'state': array.copy(),
                    'info': f'Swapped {array[j + 1]} and {array[j]}',
                    'highlight': [j, j + 1]
                })

        # Mark the last element as sorted
        frames.append({
            'state': array.copy(),
            'info': f'Element {array[n - i - 1]} is now in correct position',
            'highlight': [n - i - 1]
        })

        # If no swapping occurred in this pass, array is sorted
        if not swapped:
            break

    # Final state
    frames.append({
        'state': array.copy(),
        'info': 'Array is sorted',
        'highlight': []
    })

    return frames


def quick_sort_with_frames(arr, low, high, frames):
    """Quick sort implementation that records frames for visualization"""
    if low < high:
        # Record current state
        current_array = arr.copy()
        frames.append({
            'state': current_array,
            'info': f'Partitioning subarray from index {low} to {high}',
            'highlight': list(range(low, high + 1))
        })

        # Partition the array and get the pivot index
        pivot_index = partition(arr, low, high, frames)

        # Record the state after partitioning
        current_array = arr.copy()
        frames.append({
            'state': current_array,
            'info': f'Pivot {arr[pivot_index]} is now at correct position {pivot_index}',
            'highlight': [pivot_index]
        })

        # Recursively sort the sub-arrays
        quick_sort_with_frames(arr, low, pivot_index - 1, frames)
        quick_sort_with_frames(arr, pivot_index + 1, high, frames)

    return arr


def partition(arr, low, high, frames):
    """Partition function for quick sort that records visualization frames"""
    pivot = arr[high]
    frames.append({
        'state': arr.copy(),
        'info': f'Choosing pivot: {pivot} (index {high})',
        'highlight': [high]
    })

    i = low - 1

    for j in range(low, high):
        # Compare current element with pivot
        frames.append({
            'state': arr.copy(),
            'info': f'Comparing {arr[j]} with pivot {pivot}',
            'highlight': [j, high]
        })

        if arr[j] <= pivot:
            # Increment index of smaller element
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

            if i != j:
                frames.append({
                    'state': arr.copy(),
                    'info': f'Swapped {arr[i]} and {arr[j]}',
                    'highlight': [i, j]
                })

    # Place the pivot in its correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    if i + 1 != high:
        frames.append({
            'state': arr.copy(),
            'info': f'Placed pivot {pivot} at position {i + 1}',
            'highlight': [i + 1, high]
        })

    return i + 1


def merge_sort_with_frames(arr, frames):
    """Merge sort implementation that records frames for visualization"""
    # Add initial state
    frames.append({
        'state': arr.copy(),
        'info': 'Starting merge sort',
        'highlight': []
    })

    # Perform the sort and record frames
    return _merge_sort_impl(arr, 0, len(arr) - 1, frames)


def _merge_sort_impl(arr, start, end, frames):
    if start < end:
        # Find the middle point
        mid = (start + end) // 2

        # Record division
        frames.append({
            'state': arr.copy(),
            'info': f'Dividing array at index {mid}',
            'highlight': [mid]
        })

        # Recursively sort first and second halves
        _merge_sort_impl(arr, start, mid, frames)
        _merge_sort_impl(arr, mid + 1, end, frames)

        # Merge the sorted halves
        _merge(arr, start, mid, end, frames)

    return arr


def _merge(arr, start, mid, end, frames):
    # Create temporary arrays
    L = arr[start:mid + 1]
    R = arr[mid + 1:end + 1]

    # Record merging
    frames.append({
        'state': arr.copy(),
        'info': f'Merging subarrays from {start} to {mid} and from {mid + 1} to {end}',
        'highlight': list(range(start, end + 1))
    })

    # Initial indices of first and second subarrays
    i = j = 0

    # Initial index of merged subarray
    k = start

    while i < len(L) and j < len(R):
        # Compare elements from both arrays
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1

        k += 1

        # Record each comparison and placement
        frames.append({
            'state': arr.copy(),
            'info': f'Placing element at position {k - 1}',
            'highlight': [k - 1]
        })

    # Copy the remaining elements of L, if any
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1

        frames.append({
            'state': arr.copy(),
            'info': f'Copying remaining elements from left subarray to position {k - 1}',
            'highlight': [k - 1]
        })

    # Copy the remaining elements of R, if any
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1

        frames.append({
            'state': arr.copy(),
            'info': f'Copying remaining elements from right subarray to position {k - 1}',
            'highlight': [k - 1]
        })

    # Record the sorted subarray
    frames.append({
        'state': arr.copy(),
        'info': f'Subarray from {start} to {end} is now sorted',
        'highlight': list(range(start, end + 1))
    })