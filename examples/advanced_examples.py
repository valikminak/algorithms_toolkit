# examples/advanced_examples.py
from algorithms_toolkit.advanced.linear_programming import solve_lp
from algorithms_toolkit.advanced.approximation import (
    greedy_set_cover, greedy_vertex_cover, greedy_knapsack,
    greedy_traveling_salesman, greedy_bin_packing
)


def linear_programming_examples():
    """Example usage of linear programming algorithms."""
    # Maximize 3x + 2y subject to:
    # 2x + y <= 18
    # 2x + 3y <= 42
    # 3x + y <= 24
    # x, y >= 0

    c = [3, 2]  # Objective function coefficients
    A = [[2, 1],  # Constraint coefficients
         [2, 3],
         [3, 1]]
    b = [18, 42, 24]  # Constraint right-hand sides

    solution, value = solve_lp(c, A, b)

    print("Linear Programming Example:")
    print(f"Objective: maximize 3x + 2y")
    print(f"Constraints:")
    print(f"  2x + y <= 18")
    print(f"  2x + 3y <= 42")
    print(f"  3x + y <= 24")
    print(f"  x, y >= 0")
    print(f"Solution: {solution}")
    print(f"Optimal value: {value}")


def approximation_examples():
    """Example usage of approximation algorithms."""
    # Set Cover Example
    universe = set(range(1, 11))  # {1, 2, ..., 10}
    subsets = [
        {1, 2, 3, 8},
        {1, 2, 3, 4, 8},
        {1, 2, 3, 5},
        {2, 4, 5, 6, 7},
        {4, 5, 6, 7},
        {6, 7, 8, 9, 10},
        {3, 8, 9, 10}
    ]

    selected, ratio = greedy_set_cover(universe, subsets)

    print("Set Cover Example:")
    print(f"Universe: {universe}")
    print(f"Subsets: {subsets}")
    print(f"Selected subset indices: {selected}")
    print(f"Selected subsets: {[subsets[i] for i in selected]}")
    print(f"Approximation ratio: {ratio}")

    # Vertex Cover Example
    graph = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 3, 4],
        3: [1, 2, 4],
        4: [2, 3]
    }

    cover, ratio = greedy_vertex_cover(graph)

    print("\nVertex Cover Example:")
    print(f"Graph edges: {[(u, v) for u in graph for v in graph[u] if u < v]}")
    print(f"Vertex cover: {cover}")
    print(f"Approximation ratio: {ratio}")

    # Knapsack Example
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    selected, ratio = greedy_knapsack(values, weights, capacity)

    print("\nKnapsack Example:")
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"Capacity: {capacity}")
    print(f"Selected items: {selected}")
    print(f"Total value: {sum(values[i] for i in selected)}")
    print(f"Total weight: {sum(weights[i] for i in selected)}")
    print(f"Approximation ratio: {ratio}")

    # Bin Packing Example
    items = [0.42, 0.25, 0.27, 0.07, 0.72, 0.86, 0.09, 0.44, 0.50, 0.68]
    bin_capacity = 1.0

    bins, ratio = greedy_bin_packing(items, bin_capacity)

    print("\nBin Packing Example:")
    print(f"Items: {items}")
    print(f"Bin capacity: {bin_capacity}")
    print(f"Number of bins needed: {len(bins)}")
    for i, bin in enumerate(bins):
        bin_items = [items[j] for j in bin]
        print(f"  Bin {i + 1}: {bin_items} (sum: {sum(bin_items):.2f})")
    print(f"Approximation ratio: {ratio}")


def run_all_examples():
    """Run all advanced algorithm examples."""
    print("=" * 50)
    print("LINEAR PROGRAMMING EXAMPLES")
    print("=" * 50)
    linear_programming_examples()

    print("\n" + "=" * 50)
    print("APPROXIMATION ALGORITHM EXAMPLES")
    print("=" * 50)
    approximation_examples()


if __name__ == "__main__":
    run_all_examples()