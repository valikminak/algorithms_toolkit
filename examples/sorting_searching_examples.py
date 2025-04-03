import random
from kosmos.sorting import (
    quick_sort, merge_sort, heap_sort, insertion_sort,
    selection_sort, bubble_sort, shell_sort
)
from kosmos.sorting import (
    counting_sort, radix_sort
)
from kosmos.searching.binary import (
    binary_search, binary_search_recursive, lower_bound, upper_bound
)
from utils.performance import benchmark


def sorting_examples():
    """Example usage of sorting algorithms."""
    # Create random array
    random.seed(42)
    arr = [random.randint(1, 100) for _ in range(20)]
    arr.copy()

    print(f"Original array: {arr}")

    # Test different sorting algorithms
    sorted_arr = quick_sort(arr)
    print(f"\nQuick sort: {sorted_arr}")

    sorted_arr = merge_sort(arr)
    print(f"Merge sort: {sorted_arr}")

    sorted_arr = heap_sort(arr)
    print(f"Heap sort: {sorted_arr}")

    sorted_arr = insertion_sort(arr)
    print(f"Insertion sort: {sorted_arr}")

    sorted_arr = selection_sort(arr)
    print(f"Selection sort: {sorted_arr}")

    sorted_arr = bubble_sort(arr)
    print(f"Bubble sort: {sorted_arr}")

    sorted_arr = shell_sort(arr)
    print(f"Shell sort: {sorted_arr}")

    # Linear time sorts
    max_val = max(arr)
    sorted_arr = counting_sort(arr, max_val)
    print(f"\nCounting sort: {sorted_arr}")

    sorted_arr = radix_sort(arr)
    print(f"Radix sort: {sorted_arr}")

    # Visualize a sorting algorithm
    print("\nVisualizing Quick Sort (see plot)...")
    # Uncomment the following line to see the visualization
    # visualize_sorting_algorithm(quick_sort, original_arr, "Quick Sort")


def searching_examples():
    """Example usage of searching algorithms."""
    # Create sorted array
    arr = sorted([3, 5, 8, 10, 12, 15, 15, 15, 20, 25])
    print(f"Sorted array: {arr}")

    # Binary search
    target = 15
    index = binary_search(arr, target)
    print(f"\nBinary search for {target}: found at index {index}")

    # Recursive binary search
    index = binary_search_recursive(arr, target)
    print(f"Recursive binary search for {target}: found at index {index}")

    # Lower bound and upper bound
    lb = lower_bound(arr, target)
    ub = upper_bound(arr, target)
    print(f"Lower bound for {target}: index {lb}")
    print(f"Upper bound for {target}: index {ub}")
    print(f"Count of {target}: {ub - lb}")


def benchmark_sorting_algorithms():
    """Benchmark different sorting algorithms."""
    # List of sorting functions to benchmark
    sort_funcs = [
        quick_sort,
        merge_sort,
        heap_sort,
        insertion_sort,
        selection_sort,
        bubble_sort,
        shell_sort,
        radix_sort
    ]

    sort_names = [func.__name__ for func in sort_funcs]

    # Create input data of varying sizes
    input_sizes = [100, 500, 1000, 2000, 3000]
    inputs = []

    for size in input_sizes:
        arr = [random.randint(1, 10000) for _ in range(size)]
        inputs.append(arr)

    print("Benchmarking sorting algorithms...")
    results = benchmark(sort_funcs, inputs, sort_names)

    # Print results
    print("\nExecution times (seconds):")
    print("-" * 40)
    print(f"{'Size':<10}", end="")
    for name in sort_names:
        print(f"{name:<15}", end="")
    print()

    for i, size in enumerate(input_sizes):
        print(f"{size:<10}", end="")
        for name in sort_names:
            print(f"{results[name][i]:.6f}       ", end="")
        print()

    # Plot results
    # Uncomment to see the plot
    # plot_benchmark(results, input_sizes, "Sorting Algorithm Comparison", log_scale=True)


def run_all_examples():
    """Run all sorting and searching examples."""
    print("=" * 50)
    print("SORTING EXAMPLES")
    print("=" * 50)
    sorting_examples()

    print("\n" + "=" * 50)
    print("SEARCHING EXAMPLES")
    print("=" * 50)
    searching_examples()

    print("\n" + "=" * 50)
    print("BENCHMARK EXAMPLES")
    print("=" * 50)
    benchmark_sorting_algorithms()


if __name__ == "__main__":
    run_all_examples()