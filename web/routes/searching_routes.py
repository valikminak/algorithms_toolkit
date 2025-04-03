from flask import Blueprint, jsonify, request
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import algorithms
from kosmos.searching.binary import binary_search
from utils.performance import benchmark, measure_execution_time
from utils.matplotlib_adapter import convert_plot_to_image

searching_bp = Blueprint('searching', __name__)


@searching_bp.route('/algorithms')
def get_searching_algorithms():
    """Return available searching algorithms"""
    algorithms = [
        {"id": "binary_search", "name": "Binary Search", "complexity": "O(log n)"},
        {"id": "linear_search", "name": "Linear Search", "complexity": "O(n)"}
        # Add more algorithms as available
    ]
    return jsonify(algorithms)


def linear_search(arr, target):
    """Simple linear search implementation for comparison"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


@searching_bp.route('/run', methods=['POST'])
def run_searching():
    """Run a searching algorithm and return the results"""
    data = request.json
    algorithm_name = data.get('algorithm', 'binary_search')
    input_array = data.get('input', [1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
    target = data.get('target', 21)  # Default to searching for 21

    # Make sure input is sorted for binary search
    if algorithm_name == 'binary_search':
        input_array = sorted(input_array)

    # Map algorithm names to functions
    algorithms = {
        'binary_search': binary_search,
        'linear_search': linear_search,
    }

    algorithm = algorithms.get(algorithm_name)
    if not algorithm:
        return jsonify({'error': 'Algorithm not found'}), 404

    # Measure execution time
    result, execution_time = measure_execution_time(algorithm, input_array.copy(), target)

    # Generate visualization frames directly instead of using matplotlib
    frames = generate_search_frames(algorithm_name, input_array, target, result)

    return jsonify({
        'algorithm': algorithm_name,
        'input': input_array,
        'target': target,
        'output': result,
        'execution_time': execution_time,
        'visualization': frames,
        'category': 'searching'
    })


@searching_bp.route('/compare', methods=['POST'])
def compare_searching():
    """Compare multiple searching algorithms"""
    data = request.json
    algorithm_names = data.get('algorithms', ['binary_search', 'linear_search'])
    input_array = data.get('input', [1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
    target = data.get('target', 21)  # Default to searching for 21

    # Make sure input is sorted for binary search
    input_array = sorted(input_array)

    algorithms = {
        'binary_search': binary_search,
        'linear_search': linear_search,
    }

    # Filter out any invalid algorithm names
    selected_algorithms = [algorithms[name] for name in algorithm_names if name in algorithms]

    if not selected_algorithms:
        return jsonify({'error': 'No valid algorithms selected'}), 400

    # Create a wrapper to ensure consistent function signature for benchmark
    def create_wrapper(func):
        def wrapper(arr):
            return func(arr, target)

        return wrapper

    wrapped_algorithms = [create_wrapper(algo) for algo in selected_algorithms]

    # Run benchmark
    results = benchmark(
        wrapped_algorithms,
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


@searching_bp.route('/code', methods=['GET'])
def get_algorithm_code():
    """Return the code implementation for a searching algorithm"""
    algorithm_name = request.args.get('algorithm', 'binary_search')

    code_samples = {
        'binary_search': '''
def binary_search(arr, target):
    """
    Binary search implementation.

    Args:
        arr: A sorted array
        target: The value to search for

    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
''',
        'linear_search': '''
def linear_search(arr, target):
    """
    Linear search implementation.

    Args:
        arr: An array
        target: The value to search for

    Returns:
        Index of target if found, -1 otherwise
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
'''
    }

    return jsonify({
        'code': code_samples.get(algorithm_name, 'Code not available for this algorithm')
    })


def generate_search_frames(algorithm_name, input_array, target, result):
    """Generate visualization frames for search algorithms directly"""
    frames = []

    if algorithm_name == 'binary_search':
        # Generate binary search frames
        left, right = 0, len(input_array) - 1

        # Initial state
        frames.append({
            'state': input_array,
            'target': target,
            'highlight': [],
            'range': {'left': left, 'right': right},
            'info': 'Starting binary search'
        })

        # Search steps
        while left <= right:
            mid = (left + right) // 2
            frames.append({
                'state': input_array,
                'target': target,
                'highlight': [mid],
                'range': {'left': left, 'right': right},
                'info': f'Comparing {input_array[mid]} with target {target} at index {mid}'
            })

            if input_array[mid] == target:
                frames.append({
                    'state': input_array,
                    'target': target,
                    'highlight': [mid],
                    'range': {'left': left, 'right': right},
                    'info': f'Found target {target} at index {mid}!'
                })
                break

            if input_array[mid] < target:
                left = mid + 1
                frames.append({
                    'state': input_array,
                    'target': target,
                    'highlight': [],
                    'range': {'left': left, 'right': right},
                    'info': f'{input_array[mid]} < {target}, searching right half: [{left}...{right}]'
                })
            else:
                right = mid - 1
                frames.append({
                    'state': input_array,
                    'target': target,
                    'highlight': [],
                    'range': {'left': left, 'right': right},
                    'info': f'{input_array[mid]} > {target}, searching left half: [{left}...{right}]'
                })

        # Not found case
        if left > right and (not frames or "Found target" not in frames[-1]['info']):
            frames.append({
                'state': input_array,
                'target': target,
                'highlight': [],
                'range': {'left': left, 'right': right},
                'info': f'Target {target} not found in the array'
            })

    elif algorithm_name == 'linear_search':
        # Generate linear search frames
        frames.append({
            'state': input_array,
            'target': target,
            'highlight': [],
            'info': 'Starting linear search'
        })

        for i in range(len(input_array)):
            frames.append({
                'state': input_array,
                'target': target,
                'highlight': [i],
                'info': f'Checking index {i}: {input_array[i]} == {target}?'
            })

            if input_array[i] == target:
                frames.append({
                    'state': input_array,
                    'target': target,
                    'highlight': [i],
                    'info': f'Found {target} at index {i}!'
                })
                break

        if result == -1:
            frames.append({
                'state': input_array,
                'target': target,
                'highlight': [],
                'info': f'Target {target} not found in the array'
            })

    return frames