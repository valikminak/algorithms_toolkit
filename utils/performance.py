import time
import random
import functools
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Any, Dict, Optional, Union


def measure_execution_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure the execution time of a function.

    Args:
        func: The function to measure
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Tuple of (function_result, execution_time_in_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    return result, end_time - start_time


def benchmark(funcs: List[Callable], inputs: List[Any],
              labels: Optional[List[str]] = None, repeat: int = 3) -> Dict[str, List[float]]:
    """
    Benchmark multiple functions with multiple inputs.

    Args:
        funcs: List of functions to benchmark
        inputs: List of inputs to test with
        labels: Optional list of labels for the functions (uses function names if None)
        repeat: Number of times to repeat each measurement for averaging

    Returns:
        Dictionary mapping function labels to lists of execution times
    """
    if labels is None:
        labels = [func.__name__ for func in funcs]

    results = {label: [] for label in labels}

    for input_data in inputs:
        for func, label in zip(funcs, labels):
            # Repeat the measurement and take the average
            times = []
            for _ in range(repeat):
                _, execution_time = measure_execution_time(func, input_data)
                times.append(execution_time)

            # Store the average time
            results[label].append(sum(times) / repeat)

    return results


def plot_benchmark(results: Dict[str, List[float]], input_sizes: List[int],
                   title: str = "Algorithm Performance Comparison",
                   xlabel: str = "Input Size", ylabel: str = "Time (seconds)",
                   log_scale: bool = False, figsize: Tuple[int, int] = (10, 6)):
    """
    Plot benchmark results.

    Args:
        results: Dictionary mapping function labels to lists of execution times
        input_sizes: List of input sizes corresponding to the benchmark results
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        log_scale: Whether to use logarithmic scale for y-axis
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)

    for label, times in results.items():
        plt.plot(input_sizes, times, marker='o', label=label)

    if log_scale:
        plt.yscale('log')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def complexity_analysis(func: Callable, size_generator: Callable[[int], Any],
                        start: int = 10, end: int = 1000, step: int = 100,
                        repeat: int = 3) -> Dict[str, List[Union[int, float]]]:
    """
    Analyze the time complexity of a function empirically.

    Args:
        func: The function to analyze
        size_generator: Function that generates input of a given size
        start: Starting input size
        end: Ending input size
        step: Step size between inputs
        repeat: Number of times to repeat each measurement for averaging

    Returns:
        Dictionary with 'sizes' and 'times' lists
    """
    sizes = list(range(start, end + 1, step))
    times = []

    for size in sizes:
        # Generate input of the given size
        input_data = size_generator(size)

        # Measure execution time
        time_sum = 0
        for _ in range(repeat):
            _, execution_time = measure_execution_time(func, input_data)
            time_sum += execution_time

        times.append(time_sum / repeat)

    return {'sizes': sizes, 'times': times}


def plot_complexity(results: Dict[str, List[Union[int, float]]],
                    reference_curves: Optional[Dict[str, Callable[[int], float]]] = None,
                    title: str = "Algorithm Time Complexity",
                    xlabel: str = "Input Size", ylabel: str = "Time (seconds)",
                    log_scale: bool = False, figsize: Tuple[int, int] = (10, 6)):
    """
    Plot complexity analysis results with reference curves for common complexities.

    Args:
        results: Dictionary with 'sizes' and 'times' lists
        reference_curves: Dictionary mapping complexity names to functions
                          that calculate reference values
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        log_scale: Whether to use logarithmic scale for axes
        figsize: Figure size as (width, height) tuple
    """
    sizes = results['sizes']
    times = results['times']

    plt.figure(figsize=figsize)

    # Plot measured times
    plt.plot(sizes, times, marker='o', linestyle='-', label='Measured')

    # Plot reference curves if provided
    if reference_curves:
        # Scale the reference curves to match the measured data
        scaling_factor = times[-1] / reference_curves[list(reference_curves.keys())[0]](sizes[-1])

        for name, func in reference_curves.items():
            reference_values = [func(size) * scaling_factor for size in sizes]
            plt.plot(sizes, reference_values, linestyle='--', label=f'O({name})')

    if log_scale:
        plt.loglog()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def stress_test(func1: Callable, func2: Callable, input_generator: Callable,
                n_tests: int = 100) -> bool:
    """
    Stress test two implementations of the same algorithm against each other.

    Args:
        func1: First implementation
        func2: Second implementation
        input_generator: Function that generates random input
        n_tests: Number of tests to run

    Returns:
        True if all tests pass, False otherwise
    """
    for i in range(n_tests):
        input_data = input_generator()

        try:
            result1 = func1(input_data)
            result2 = func2(input_data)

            if result1 != result2:
                print(f"Test failed on input: {input_data}")
                print(f"func1 output: {result1}")
                print(f"func2 output: {result2}")
                return False

        except Exception as e:
            print(f"Exception on test {i} with input: {input_data}")
            print(f"Exception: {e}")
            return False

    print(f"All {n_tests} tests passed!")
    return True


# Common reference complexity functions
def common_complexity_references() -> Dict[str, Callable[[int], float]]:
    """
    Get common reference complexity functions for plotting.

    Returns:
        Dictionary mapping complexity names to functions
    """
    return {
        "1": lambda n: 1,  # O(1)
        "log n": lambda n: max(1, n.bit_length()),  # O(log n)
        "n": lambda n: n,  # O(n)
        "n log n": lambda n: n * max(1, n.bit_length()),  # O(n log n)
        "n²": lambda n: n * n,  # O(n²)
        "n³": lambda n: n * n * n,  # O(n³)
        "2^n": lambda n: 2 ** min(n, 30)  # O(2^n), limited to avoid overflow
    }