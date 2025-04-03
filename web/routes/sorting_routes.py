from flask import Blueprint, jsonify, request
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import your algorithms
from sorting.comparison import bubble_sort, quick_sort, merge_sort
from utils.visualization import visualize_sorting_algorithm
from utils.performance import benchmark, measure_execution_time
from adapters.matplotlib_adapter import convert_plot_to_image, convert_animation_to_frames

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

    # Create animation using your visualization utility
    # Convert the animation to a sequence of frames for the web
    try:
        animation = visualize_sorting_algorithm(algorithm, input_array.copy())
        frames = convert_animation_to_frames(animation)
    except Exception:
        # Fallback to simpler visualization if animation fails
        frames = [
            {'state': input_array, 'info': 'Initial array'},
            {'state': result, 'info': 'Sorted array'}
        ]

    return jsonify({
        'algorithm': algorithm_name,
        'input': input_array,
        'output': result,
        'execution_time': execution_time,
        'visualization': frames
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