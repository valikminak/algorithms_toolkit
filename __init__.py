# __init__.py
from typing import Any, List, Dict, Tuple, Set, Optional, Callable, Union, Generic, TypeVar

__version__ = "1.0.0"

# Import all submodules to make them available from the package
from graph import *
from tree import *
from string import *
from dp import *
from sorting import *
from searching import *
from geometry import *
from advanced import *
from utils import *


# Example functions to showcase usage
def graph_example():
    """Run an example of graph """
    from graph.base import Graph
    from graph.traversal import breadth_first_search, depth_first_search
    from graph.shortest_path import dijkstra
    from graph.mst import kruskal_mst

    # Create a weighted, undirected graph
    g = Graph(directed=False, weighted=True)

    # Add vertices
    for i in range(6):
        g.add_vertex(i)

    # Add edges
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 2)
    g.add_edge(1, 2, 5)
    g.add_edge(1, 3, 10)
    g.add_edge(2, 3, 3)
    g.add_edge(2, 4, 8)
    g.add_edge(3, 4, 2)
    g.add_edge(3, 5, 7)
    g.add_edge(4, 5, 6)

    print("Graph:")
    print(g)

    # Run BFS
    print("\nBFS from vertex 0:")
    bfs_result = breadth_first_search(g, 0)
    print(bfs_result)

    # Run DFS
    print("\nDFS from vertex 0:")
    dfs_result = depth_first_search(g, 0)
    print(dfs_result)

    # Run Dijkstra's algorithm
    print("\nDijkstra's algorithm from vertex 0:")
    distances, predecessors = dijkstra(g, 0)

    # Display paths and distances
    for i in range(6):
        path = []
        current = i
        while current is not None:
            path.append(current)
            current = predecessors.get(current)

        path.reverse()
        print(f"Path to {i}: {path}, Distance: {distances[i]}")

    # Find MST
    print("\nMinimum Spanning Tree:")
    mst = kruskal_mst(g)
    print(mst)


def tree_example():
    """Run an example of tree """
    from tree.base import BinaryTreeNode
    from tree.traversal import binary_tree_inorder_traversal, binary_tree_levelorder_traversal
    from tree.properties import binary_tree_height, binary_tree_is_balanced
    from tree.trie import Trie

    # Create a binary tree
    root = BinaryTreeNode(1)
    root.left = BinaryTreeNode(2)
    root.right = BinaryTreeNode(3)
    root.left.left = BinaryTreeNode(4)
    root.left.right = BinaryTreeNode(5)
    root.right.left = BinaryTreeNode(6)

    print("Binary Tree Traversals:")
    print(f"Inorder: {binary_tree_inorder_traversal(root)}")
    print(f"Level-order: {binary_tree_levelorder_traversal(root)}")

    print(f"\nTree Height: {binary_tree_height(root)}")
    print(f"Is Balanced: {binary_tree_is_balanced(root)}")

    # Create a trie
    trie = Trie()
    words = ["apple", "app", "application", "banana", "band", "bandage"]

    for word in words:
        trie.insert(word)

    print("\nTrie Operations:")
    print(f"Search 'app': {trie.search('app')}")
    print(f"Search 'appl': {trie.search('appl')}")
    print(f"Starts with 'app': {trie.starts_with('app')}")
    print(f"Words with prefix 'app': {trie.get_words_with_prefix('app')}")


def string_example():
    """Run an example of string """
    from string.pattern_matching import kmp_search, rabin_karp_search
    from string.palindrome import longest_palindromic_substring, manacher_algorithm
    from string.sequence import longest_common_subsequence, edit_distance

    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"

    print("String Pattern Matching:")
    print(f"KMP Search: {kmp_search(text, pattern)}")
    print(f"Rabin-Karp Search: {rabin_karp_search(text, pattern)}")

    s = "babad"
    print(f"\nLongest Palindromic Substring of '{s}': {longest_palindromic_substring(s)}")
    print(f"Using Manacher's Algorithm: {manacher_algorithm(s)}")

    s1 = "ABCBDAB"
    s2 = "BDCABA"
    print(f"\nLongest Common Subsequence of '{s1}' and '{s2}': {longest_common_subsequence(s1, s2)}")

    s1 = "kitten"
    s2 = "sitting"
    print(f"Edit Distance between '{s1}' and '{s2}': {edit_distance(s1, s2)}")


def dp_example():
    """Run an example of dynamic programming """
    from dp.classic import knapsack_01_with_solution, rod_cutting_with_solution, coin_change
    from dp.sequence import longest_increasing_subsequence, longest_palindromic_subsequence

    # Knapsack problem
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    max_value, selected_items = knapsack_01_with_solution(values, weights, capacity)
    print("0/1 Knapsack Problem:")
    print(f"Maximum Value: {max_value}")
    print(f"Selected Items: {selected_items}")

    # Rod cutting problem
    prices = [1, 5, 8, 9, 10, 17, 17, 20]
    rod_length = 8

    max_revenue, cuts = rod_cutting_with_solution(prices, rod_length)
    print("\nRod Cutting Problem:")
    print(f"Maximum Revenue: {max_revenue}")
    print(f"Optimal Cuts: {cuts}")

    # Coin change problem
    coins = [1, 2, 5]
    amount = 11

    min_coins = coin_change(coins, amount)
    print("\nCoin Change Problem:")
    print(f"Minimum Coins Needed for {amount}: {min_coins}")

    # Longest Increasing Subsequence
    nums = [10, 22, 9, 33, 21, 50, 41, 60, 80]
    lis = longest_increasing_subsequence(nums)
    print("\nLongest Increasing Subsequence:")
    print(f"Original Sequence: {nums}")
    print(f"LIS: {lis}")


def sorting_example():
    """Run an example of sorting """
    from sorting.linear import counting_sort, radix_sort
    import random

    # Create a random array
    arr = [random.randint(1, 100) for _ in range(20)]
    print(f"Original Array: {arr}")

    # Test sorting algorithms
    print(f"\nQuick Sort: {quick_sort(arr.copy())}")
    print(f"Merge Sort: {merge_sort(arr.copy())}")
    print(f"Heap Sort: {heap_sort(arr.copy())}")
    print(f"Counting Sort: {counting_sort(arr.copy())}")
    print(f"Radix Sort: {radix_sort(arr.copy())}")


def searching_example():
    """Run an example of searching """
    from searching.binary import binary_search, lower_bound, upper_bound

    # Create a sorted array
    arr = sorted([3, 5, 8, 10, 12, 15, 15, 15, 20, 25])
    print(f"Sorted Array: {arr}")

    # Test binary search
    target = 15
    print(f"\nBinary Search for {target}: {binary_search(arr, target)}")

    # Test lower and upper bound
    print(f"Lower Bound for {target}: {lower_bound(arr, target)}")
    print(f"Upper Bound for {target}: {upper_bound(arr, target)}")


def geometry_example():
    """Run an example of geometric """
    from geometry.basic import Point
    from geometry.convex_hull import convex_hull
    from geometry.intersection import line_segment_intersection
    import random

    # Create random points
    points = [Point(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
    print("Points:")
    for i, p in enumerate(points):
        print(f"Point {i}: ({p.x}, {p.y})")

    # Compute convex hull
    hull = convex_hull(points)
    print("\nConvex Hull Points:")
    for p in hull:
        print(f"({p.x}, {p.y})")

    # Test line intersection
    line1 = (Point(10, 10), Point(50, 50))
    line2 = (Point(10, 50), Point(50, 10))

    intersection = line_segment_intersection(line1[0], line1[1], line2[0], line2[1])
    print("\nLine Segment Intersection:")
    if intersection:
        print(f"Intersect at ({intersection.x}, {intersection.y})")
    else:
        print("No intersection")


def advanced_example():
    """Run an example of advanced """
    from advanced.linear_programming import solve_lp
    from advanced.approximation import greedy_set_cover, greedy_vertex_cover

    # Linear Programming Example
    c = [3, 2]  # Maximize 3x + 2y
    A = [[2, 1],  # 2x + y <= 18
         [2, 3],  # 2x + 3y <= 42
         [3, 1]]  # 3x + y <= 24
    b = [18, 42, 24]

    solution, value = solve_lp(c, A, b)
    print("Linear Programming Example:")
    print(f"Optimal Solution: {solution}")
    print(f"Optimal Value: {value}")

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
    print("\nSet Cover Example:")
    print(f"Selected Subsets: {selected}")
    print(f"Approximation Ratio: {ratio}")

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
    print(f"Vertex Cover: {cover}")
    print(f"Approximation Ratio: {ratio}")


def run_all_examples():
    """Run all algorithm examples."""
    print("=" * 50)
    print("GRAPH ALGORITHMS")
    print("=" * 50)
    graph_example()

    print("\n" + "=" * 50)
    print("TREE ALGORITHMS")
    print("=" * 50)
    tree_example()

    print("\n" + "=" * 50)
    print("STRING ALGORITHMS")
    print("=" * 50)
    string_example()

    print("\n" + "=" * 50)
    print("DYNAMIC PROGRAMMING ALGORITHMS")
    print("=" * 50)
    dp_example()

    print("\n" + "=" * 50)
    print("SORTING ALGORITHMS")
    print("=" * 50)
    sorting_example()

    print("\n" + "=" * 50)
    print("SEARCHING ALGORITHMS")
    print("=" * 50)
    searching_example()

    print("\n" + "=" * 50)
    print("GEOMETRIC ALGORITHMS")
    print("=" * 50)
    geometry_example()

    print("\n" + "=" * 50)
    print("ADVANCED ALGORITHMS")
    print("=" * 50)
    advanced_example()


if __name__ == "__main__":
    run_all_examples()
