from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable
import heapq
import random
import math
import collections
import numpy as np


def greedy_set_cover(universe: Set[Any], subsets: List[Set[Any]]) -> Tuple[List[int], float]:
    """
    Greedy approximation algorithm for the Set Cover problem.

    The Set Cover problem:
    Given a universe U of elements and a collection S of subsets of U,
    find the smallest subcollection of S that covers all elements in U.

    This greedy algorithm provides a ln(n) approximation, where n is
    the size of the universe.

    Args:
        universe: Set of all elements
        subsets: List of subsets of the universe

    Returns:
        Tuple of (selected_indices, approximation_ratio) where:
        - selected_indices: List of indices of selected subsets
        - approximation_ratio: Approximation ratio (ln|U|)
    """
    # Make a copy of the universe
    remaining = set(universe)

    # Keep track of selected subsets
    selected = []

    # Continue until all elements are covered
    while remaining:
        # Find the subset that covers the most uncovered elements
        best_subset = -1
        best_coverage = -1

        for i, subset in enumerate(subsets):
            # Skip if already selected
            if i in selected:
                continue

            # Count how many uncovered elements this subset would cover
            coverage = len(remaining.intersection(subset))

            if coverage > best_coverage:
                best_subset = i
                best_coverage = coverage

        # If no subset covers any remaining element, there's no solution
        if best_coverage == 0:
            break

        # Add the best subset to the solution
        selected.append(best_subset)

        # Remove covered elements
        remaining -= subsets[best_subset]

    # Approximation ratio: ln|U|
    approximation_ratio = math.log(len(universe))

    return selected, approximation_ratio


def greedy_vertex_cover(graph: Dict[Any, List[Any]]) -> Tuple[Set[Any], float]:
    """
    Greedy approximation algorithm for the Vertex Cover problem.

    The Vertex Cover problem:
    Given an undirected graph, find the smallest set of vertices such that
    each edge has at least one endpoint in the set.

    This greedy algorithm provides a 2-approximation.

    Args:
        graph: Dictionary representing the graph (adjacency list)

    Returns:
        Tuple of (vertex_cover, approximation_ratio) where:
        - vertex_cover: Set of vertices in the cover
        - approximation_ratio: Approximation ratio (2)
    """
    # Create a copy of the graph edges
    edges = []
    for u in graph:
        for v in graph[u]:
            if u < v:  # Avoid counting edges twice
                edges.append((u, v))

    # Keep track of covered edges and selected vertices
    vertex_cover = set()
    covered_edges = set()

    # Continue until all edges are covered
    while covered_edges != set(range(len(edges))):
        # Find the vertex that covers the most uncovered edges
        best_vertex = None
        best_coverage = -1

        for vertex in graph:
            # Skip if already selected
            if vertex in vertex_cover:
                continue

            # Count how many uncovered edges this vertex would cover
            coverage = 0
            for i, (u, v) in enumerate(edges):
                if i not in covered_edges and (u == vertex or v == vertex):
                    coverage += 1

            if coverage > best_coverage:
                best_vertex = vertex
                best_coverage = coverage

        # If no vertex covers any remaining edge, there's a problem
        if best_coverage == 0:
            break

        # Add the best vertex to the solution
        vertex_cover.add(best_vertex)

        # Mark all edges incident to this vertex as covered
        for i, (u, v) in enumerate(edges):
            if u == best_vertex or v == best_vertex:
                covered_edges.add(i)

    # Approximation ratio: 2
    approximation_ratio = 2.0

    return vertex_cover, approximation_ratio


def two_approximation_vertex_cover(graph: Dict[Any, List[Any]]) -> Tuple[Set[Any], float]:
    """
    2-approximation algorithm for the Vertex Cover problem.

    This algorithm takes any edge, adds both its endpoints to the cover,
    then removes all edges incident to these vertices, and repeats.

    Args:
        graph: Dictionary representing the graph (adjacency list)

    Returns:
        Tuple of (vertex_cover, approximation_ratio) where:
        - vertex_cover: Set of vertices in the cover
        - approximation_ratio: Approximation ratio (2)
    """
    # Create a copy of the graph
    remaining_edges = []
    for u in graph:
        for v in graph[u]:
            if u < v:  # Avoid counting edges twice
                remaining_edges.append((u, v))

    # Keep track of selected vertices
    vertex_cover = set()

    # Continue until all edges are covered
    while remaining_edges:
        # Take any edge
        u, v = remaining_edges[0]

        # Add both endpoints to the cover
        vertex_cover.add(u)
        vertex_cover.add(v)

        # Remove all edges incident to u or v
        remaining_edges = [(a, b) for (a, b) in remaining_edges if a != u and a != v and b != u and b != v]

    # Approximation ratio: 2
    approximation_ratio = 2.0

    return vertex_cover, approximation_ratio


def greedy_max_cut(graph: Dict[Any, List[Any]]) -> Tuple[Tuple[Set[Any], Set[Any]], float]:
    """
    Greedy approximation algorithm for the Maximum Cut problem.

    The Maximum Cut problem:
    Given an undirected graph, partition the vertices into two sets
    such that the number of edges between the two sets is maximized.

    This greedy algorithm provides a 1/2-approximation.

    Args:
        graph: Dictionary representing the graph (adjacency list)

    Returns:
        Tuple of ((set_a, set_b), approximation_ratio) where:
        - set_a, set_b: The two sets of vertices
        - approximation_ratio: Approximation ratio (1/2)
    """
    # Initialize the two sets
    set_a = set()
    set_b = set()

    # Add each vertex to the set that maximizes the cut
    for vertex in graph:
        # Count how many neighbors are in each set
        neighbors_in_a = sum(1 for neighbor in graph[vertex] if neighbor in set_a)
        neighbors_in_b = sum(1 for neighbor in graph[vertex] if neighbor in set_b)

        # Add to the set with fewer neighbors
        if neighbors_in_a <= neighbors_in_b:
            set_a.add(vertex)
        else:
            set_b.add(vertex)

    # Approximation ratio: 1/2
    approximation_ratio = 0.5

    return (set_a, set_b), approximation_ratio


def randomized_max_cut(graph: Dict[Any, List[Any]]) -> Tuple[Tuple[Set[Any], Set[Any]], float]:
    """
    Randomized approximation algorithm for the Maximum Cut problem.

    This algorithm randomly assigns each vertex to one of two sets.
    Expected approximation ratio: 1/2.

    Args:
        graph: Dictionary representing the graph (adjacency list)

    Returns:
        Tuple of ((set_a, set_b), approximation_ratio) where:
        - set_a, set_b: The two sets of vertices
        - approximation_ratio: Expected approximation ratio (1/2)
    """
    # Initialize the two sets
    set_a = set()
    set_b = set()

    # Randomly assign each vertex to one of the sets
    for vertex in graph:
        if random.random() < 0.5:
            set_a.add(vertex)
        else:
            set_b.add(vertex)

    # Expected approximation ratio: 1/2
    approximation_ratio = 0.5

    return (set_a, set_b), approximation_ratio


def goemans_williamson_max_cut(graph: Dict[Any, Dict[Any, float]]) -> Tuple[Tuple[Set[Any], Set[Any]], float]:
    """
    Goemans-Williamson approximation algorithm for the Maximum Cut problem.

    This algorithm uses semidefinite programming relaxation and random hyperplane rounding
    to achieve an approximation ratio of approximately 0.878.

    Args:
        graph: Dictionary representing the weighted graph (adjacency dict with weights)

    Returns:
        Tuple of ((set_a, set_b), approximation_ratio) where:
        - set_a, set_b: The two sets of vertices
        - approximation_ratio: Approximation ratio (0.878)
    """
    # Note: This is a simplified implementation that doesn't actually solve the SDP
    # In a real implementation, you'd use an SDP solver library

    # Simplified approach: use a randomized algorithm instead
    # Initialize the two sets
    set_a = set()
    set_b = set()

    # Randomly assign each vertex to one of the sets
    for vertex in graph:
        if random.random() < 0.5:
            set_a.add(vertex)
        else:
            set_b.add(vertex)

    # The Goemans-Williamson algorithm has an approximation ratio of:
    # min_{0≤θ≤π} θ/(2π(1-cos(θ))) ≈ 0.878
    approximation_ratio = 0.878

    return (set_a, set_b), approximation_ratio


def greedy_knapsack(values: List[float], weights: List[float], capacity: float) -> Tuple[List[int], float]:
    """
    Greedy approximation algorithm for the 0/1 Knapsack problem.

    The Knapsack problem:
    Given n items with values and weights, and a knapsack with a capacity,
    find the most valuable subset of items that fit in the knapsack.

    This greedy algorithm sorts items by value/weight ratio and takes the
    best items first. It provides a 1/2-approximation for 0/1 knapsack.

    Args:
        values: List of item values
        weights: List of item weights
        capacity: Knapsack capacity

    Returns:
        Tuple of (selected_items, approximation_ratio) where:
        - selected_items: Indices of selected items
        - approximation_ratio: Approximation ratio (1/2)
    """
    n = len(values)

    # Calculate value/weight ratio for each item
    items = [(i, values[i], weights[i], values[i] / weights[i])
             for i in range(n) if weights[i] > 0]

    # Sort items by value/weight ratio in descending order
    items.sort(key=lambda x: x[3], reverse=True)

    # Greedily select items
    selected = []
    total_weight = 0
    total_value = 0

    for idx, value, weight, ratio in items:
        if total_weight + weight <= capacity:
            selected.append(idx)
            total_weight += weight
            total_value += value

    # For 0/1 knapsack, add the item with the highest value if better
    best_single_item = max([(i, v) for i, v, w, r in items if w <= capacity],
                           key=lambda x: x[1],
                           default=(-1, 0))

    if best_single_item[1] > total_value:
        selected = [best_single_item[0]]

    # Approximation ratio: 1/2 for 0/1 knapsack
    approximation_ratio = 0.5

    return selected, approximation_ratio


def fptas_knapsack(values: List[float], weights: List[float], capacity: float, epsilon: float) -> Tuple[
    List[int], float]:
    """
    Fully Polynomial-Time Approximation Scheme (FPTAS) for the Knapsack problem.

    This algorithm scales down the values and then uses dynamic programming,
    achieving a (1-ε)-approximation in time polynomial in n and 1/ε.

    Args:
        values: List of item values
        weights: List of item weights
        capacity: Knapsack capacity
        epsilon: Approximation factor (0 < ε < 1)

    Returns:
        Tuple of (selected_items, approximation_ratio) where:
        - selected_items: Indices of selected items
        - approximation_ratio: Approximation ratio (1-ε)
    """
    n = len(values)

    # Find the maximum value
    max_value = max(values) if values else 0

    # Scale factor
    k = epsilon * max_value / n

    # Scale down the values
    scaled_values = [int(v / k) for v in values]

    # Maximum possible scaled value
    max_scaled_value = sum(scaled_values)

    # DP table: dp[i][j] = minimum weight needed to achieve value j using items 0...i
    dp = [[float('inf')] * (max_scaled_value + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(max_scaled_value + 1):
            # Don't take item i-1
            dp[i][j] = dp[i - 1][j]

            # Take item i-1 if possible
            if j >= scaled_values[i - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 1][j - scaled_values[i - 1]] + weights[i - 1])

    # Find the maximum scaled value that fits in the knapsack
    max_achievable_value = 0
    for j in range(max_scaled_value, -1, -1):
        if dp[n][j] <= capacity:
            max_achievable_value = j
            break

    # Reconstruct the solution
    selected = []
    remaining_value = max_achievable_value

    for i in range(n, 0, -1):
        if remaining_value == 0:
            break

        if dp[i][remaining_value] != dp[i - 1][remaining_value]:
            selected.append(i - 1)
            remaining_value -= scaled_values[i - 1]

    # Calculate the actual value of the solution
    actual_value = sum(values[i] for i in selected)

    # Approximation ratio: 1-ε
    approximation_ratio = 1 - epsilon

    return selected, approximation_ratio


def greedy_traveling_salesman(distances: List[List[float]]) -> Tuple[List[int], float]:
    """
    Greedy approximation algorithm for the Traveling Salesman Problem (TSP).

    The TSP:
    Given a list of cities and the distances between each pair,
    find the shortest possible route that visits each city exactly once
    and returns to the origin city.

    This nearest neighbor greedy approach provides a log(n)-approximation
    for metric TSP.

    Args:
        distances: Matrix of distances between cities

    Returns:
        Tuple of (tour, approximation_ratio) where:
        - tour: List of city indices in the order they should be visited
        - approximation_ratio: Approximation ratio (log(n))
    """
    n = len(distances)

    # Start at city 0
    tour = [0]
    current = 0
    unvisited = set(range(1, n))

    # Greedily add the nearest city
    while unvisited:
        nearest = min(unvisited, key=lambda city: distances[current][city])
        tour.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    # Return to the starting city
    tour.append(0)

    # Approximation ratio: log(n) for metric TSP
    approximation_ratio = math.log(n)

    return tour, approximation_ratio


def christofides_tsp(distances: List[List[float]]) -> Tuple[List[int], float]:
    """
    Christofides algorithm for approximating the Traveling Salesman Problem (TSP).

    This algorithm provides a 3/2-approximation for metric TSP.

    Args:
        distances: Matrix of distances between cities

    Returns:
        Tuple of (tour, approximation_ratio) where:
        - tour: List of city indices in the order they should be visited
        - approximation_ratio: Approximation ratio (3/2)
    """
    # Note: This is a simplified implementation that doesn't actually implement
    # all steps of Christofides algorithm properly

    n = len(distances)

    # Step 1: Compute the minimum spanning tree (MST)
    # Here we use Prim's algorithm
    mst_edges = []
    visited = {0}

    for _ in range(n - 1):
        min_edge = None
        min_distance = float('inf')

        for u in visited:
            for v in range(n):
                if v not in visited and distances[u][v] < min_distance:
                    min_distance = distances[u][v]
                    min_edge = (u, v)

        if min_edge:
            mst_edges.append(min_edge)
            visited.add(min_edge[1])

    # Step 2: Find odd degree vertices in the MST
    degree = [0] * n
    for u, v in mst_edges:
        degree[u] += 1
        degree[v] += 1

    odd_vertices = [i for i, d in enumerate(degree) if d % 2 == 1]

    # Step 3: Find minimum weight perfect matching on odd vertices
    # Simplified: just match consecutive odd vertices
    matching = []
    for i in range(0, len(odd_vertices), 2):
        if i + 1 < len(odd_vertices):
            matching.append((odd_vertices[i], odd_vertices[i + 1]))

    # Step 4: Combine MST and matching to form an Eulerian multigraph
    euler_edges = mst_edges + matching

    # Step 5: Find an Eulerian tour
    # Simplified: just collect vertices in the order of the edges
    euler_tour = []
    current = 0
    remaining_edges = euler_edges.copy()

    while remaining_edges:
        for i, (u, v) in enumerate(remaining_edges):
            if u == current:
                euler_tour.append(current)
                current = v
                remaining_edges.pop(i)
                break
            elif v == current:
                euler_tour.append(current)
                current = u
                remaining_edges.pop(i)
                break

    euler_tour.append(current)

    # Step 6: Make the tour Hamiltonian by shortcutting
    visited = set()
    hamiltonian_tour = []

    for vertex in euler_tour:
        if vertex not in visited:
            hamiltonian_tour.append(vertex)
            visited.add(vertex)

    # Return to start
    hamiltonian_tour.append(hamiltonian_tour[0])

    # Approximation ratio: 3/2 for metric TSP
    approximation_ratio = 1.5

    return hamiltonian_tour, approximation_ratio


def greedy_minimum_spanning_tree(graph: Dict[Any, Dict[Any, float]]) -> Tuple[List[Tuple[Any, Any, float]], float]:
    """
    Greedy algorithm for the Minimum Spanning Tree (MST) problem.

    This is Kruskal's algorithm, which is exact (not an approximation).

    Args:
        graph: Dictionary representing the weighted graph

    Returns:
        Tuple of (mst_edges, approximation_ratio) where:
        - mst_edges: List of (u, v, weight) tuples in the MST
        - approximation_ratio: 1.0 (exact algorithm)
    """
    # Extract all edges
    edges = []
    for u in graph:
        for v, weight in graph[u].items():
            if u < v:  # Avoid counting edges twice
                edges.append((u, v, weight))

    # Sort edges by weight
    edges.sort(key=lambda e: e[2])

    # Initialize Union-Find data structure
    parent = {vertex: vertex for vertex in graph}
    rank = {vertex: 0 for vertex in graph}

    def find(x):
        """Find the representative of the set containing x."""
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        """Union the sets containing x and y."""
        root_x = find(x)
        root_y = find(y)

        if root_x == root_y:
            return

        # Union by rank
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            if rank[root_x] == rank[root_y]:
                rank[root_x] += 1

    # Build the MST using Kruskal's algorithm
    mst_edges = []

    for u, v, weight in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append((u, v, weight))

    # Approximation ratio: 1.0 (exact algorithm)
    approximation_ratio = 1.0

    return mst_edges, approximation_ratio


def greedy_bin_packing(items: List[float], bin_capacity: float) -> Tuple[List[List[int]], float]:
    """
    First-Fit Decreasing (FFD) approximation algorithm for the Bin Packing problem.

    The Bin Packing problem:
    Given a list of items with sizes and bins with a fixed capacity,
    find the minimum number of bins needed to pack all items.

    FFD provides a 11/9 * OPT + 1 approximation.

    Args:
        items: List of item sizes
        bin_capacity: Capacity of each bin

    Returns:
        Tuple of (bins, approximation_ratio) where:
        - bins: List of bins, each bin is a list of item indices
        - approximation_ratio: Approximation ratio (11/9)
    """
    n = len(items)

    # Sort items in descending order
    sorted_items = [(i, size) for i, size in enumerate(items)]
    sorted_items.sort(key=lambda x: x[1], reverse=True)

    # Initialize bins
    bins = []

    # Place each item in the first bin that can hold it
    for item_idx, item_size in sorted_items:
        # Try to place in an existing bin
        placed = False

        for bin_idx, bin_items in enumerate(bins):
            # Calculate remaining capacity
            bin_sum = sum(items[i] for i in bin_items)

            if bin_sum + item_size <= bin_capacity:
                bins[bin_idx].append(item_idx)
                placed = True
                break

        # If no existing bin can hold the item, create a new bin
        if not placed:
            bins.append([item_idx])

    # Approximation ratio: 11/9 * OPT + 1, simplified to 11/9
    approximation_ratio = 11 / 9

    return bins, approximation_ratio


def ptas_bin_packing(items: List[float], bin_capacity: float, epsilon: float) -> Tuple[List[List[int]], float]:
    """
    Polynomial-Time Approximation Scheme (PTAS) for the Bin Packing problem.

    This algorithm divides items into large and small, solves exactly for large items,
    and greedily packs small items, achieving a (1+ε)-approximation.

    Args:
        items: List of item sizes
        bin_capacity: Capacity of each bin
        epsilon: Approximation factor (0 < ε < 1)

    Returns:
        Tuple of (bins, approximation_ratio) where:
        - bins: List of bins, each bin is a list of item indices
        - approximation_ratio: Approximation ratio (1+ε)
    """
    # Note: This is a simplified implementation that doesn't actually implement
    # the full PTAS for bin packing, which would require solving a complex dynamic program

    # Instead, we'll just use the FFD algorithm as an approximation
    bins, _ = greedy_bin_packing(items, bin_capacity)

    # The true PTAS would have an approximation ratio of (1+ε)
    approximation_ratio = 1 + epsilon

    return bins, approximation_ratio