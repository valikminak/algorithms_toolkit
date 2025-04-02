from typing import Dict, Any
import collections
import random


def greedy_coloring(graph) -> Dict[Any, int]:
    """
    Greedy graph coloring algorithm.

    This algorithm assigns a color to each vertex such that no two adjacent vertices
    have the same color, using the smallest possible color index for each vertex.

    Args:
        graph: The graph to color

    Returns:
        Dictionary mapping each vertex to its color (integer starting from 0)
    """
    if not graph.vertices:
        return {}

    # Initialize result for all vertices as unassigned
    color_map = {}

    # Assign the first color (0) to the first vertex
    # Then process remaining vertices one by one
    for vertex in graph.vertices:
        # Create a set of colors that are already assigned to adjacent vertices
        adjacent_colors = {color_map.get(neighbor) for neighbor in graph.get_neighbors(vertex) if neighbor in color_map}

        # Find the first available color
        color = 0
        while color in adjacent_colors:
            color += 1

        # Assign the found color
        color_map[vertex] = color

    return color_map


def dsatur_coloring(graph) -> Dict[Any, int]:
    """
    DSATUR graph coloring algorithm.

    This algorithm prioritizes vertices with more colors in their neighborhood
    (degree of saturation), which usually results in fewer colors than greedy coloring.

    Args:
        graph: The graph to color

    Returns:
        Dictionary mapping each vertex to its color (integer starting from 0)
    """
    if not graph.vertices:
        return {}

    # Initialize coloring and saturation degree
    color_map = {}
    saturation = {v: 0 for v in graph.vertices}
    uncolored = set(graph.vertices)

    # Start with vertex of highest degree
    first_vertex = max(graph.vertices, key=lambda v: len(graph.get_neighbors(v)))
    color_map[first_vertex] = 0
    uncolored.remove(first_vertex)

    # Update saturation of neighbors
    for neighbor in graph.get_neighbors(first_vertex):
        saturation[neighbor] += 1

    # Process remaining vertices
    while uncolored:
        # Select vertex with highest saturation degree
        # Break ties by selecting the vertex with highest degree
        next_vertex = max(uncolored,
                          key=lambda v: (saturation[v], len(graph.get_neighbors(v))))

        # Find the smallest available color
        used_colors = {color_map[neighbor] for neighbor in graph.get_neighbors(next_vertex) if neighbor in color_map}
        color = 0
        while color in used_colors:
            color += 1

        # Assign color to the vertex
        color_map[next_vertex] = color
        uncolored.remove(next_vertex)

        # Update saturation of uncolored neighbors
        for neighbor in graph.get_neighbors(next_vertex):
            if neighbor in uncolored:
                # Check if the color we just used is new to this neighbor
                if all(color_map.get(n) != color for n in graph.get_neighbors(neighbor) if
                       n in color_map and n != next_vertex):
                    saturation[neighbor] += 1

    return color_map


def recursive_largest_first_coloring(graph) -> Dict[Any, int]:
    """
    Recursive Largest First (RLF) graph coloring algorithm.

    This algorithm builds each color class (set of vertices with the same color)
    one at a time, prioritizing vertices with many uncolored neighbors.

    Args:
        graph: The graph to color

    Returns:
        Dictionary mapping each vertex to its color (integer starting from 0)
    """
    if not graph.vertices:
        return {}

    # Initialize coloring
    color_map = {}
    uncolored = set(graph.vertices)
    color = 0

    while uncolored:
        # Set of vertices that can be colored with current color
        candidates = set(uncolored)

        # Build the next color class
        while candidates:
            # Select vertex with most uncolored neighbors
            next_vertex = max(candidates,
                              key=lambda v: sum(1 for n in graph.get_neighbors(v) if n in uncolored))

            # Assign color and update sets
            color_map[next_vertex] = color
            uncolored.remove(next_vertex)

            # Remove neighbors from candidates
            candidates -= set(graph.get_neighbors(next_vertex))

        color += 1

    return color_map


def welsh_powell_coloring(graph) -> Dict[Any, int]:
    """
    Welsh-Powell graph coloring algorithm.

    This algorithm sorts vertices by degree and colors them in that order,
    skipping vertices that would create conflicts.

    Args:
        graph: The graph to color

    Returns:
        Dictionary mapping each vertex to its color (integer starting from 0)
    """
    if not graph.vertices:
        return {}

    # Sort vertices by degree in descending order
    sorted_vertices = sorted(graph.vertices,
                             key=lambda v: len(graph.get_neighbors(v)),
                             reverse=True)

    # Initialize coloring
    color_map = {}

    # Process vertices in order of degree
    for vertex in sorted_vertices:
        if vertex in color_map:
            continue

        # Find colors used by neighbors
        used_colors = {color_map.get(neighbor) for neighbor in graph.get_neighbors(vertex) if neighbor in color_map}

        # Find the smallest available color
        color = 0
        while color in used_colors:
            color += 1

        # Assign this color to the vertex
        color_map[vertex] = color

        # Try to assign the same color to other uncolored vertices
        for v in sorted_vertices:
            if v not in color_map:
                # Check if any neighbor of v has already been assigned this color
                if not any(neighbor in color_map and color_map[neighbor] == color
                           for neighbor in graph.get_neighbors(v)):
                    color_map[v] = color

    return color_map


def tabu_search_coloring(graph, max_iterations=1000, tabu_tenure=10) -> Dict[Any, int]:
    """
    Tabu Search algorithm for graph coloring.

    This is a metaheuristic that iteratively improves a coloring by making
    local moves, avoiding recently visited solutions (tabu).

    Args:
        graph: The graph to color
        max_iterations: Maximum number of iterations
        tabu_tenure: Number of iterations a move is considered tabu

    Returns:
        Dictionary mapping each vertex to its color (integer starting from 0)
    """
    if not graph.vertices:
        return {}

    # Get initial coloring using greedy algorithm
    color_map = greedy_coloring(graph)

    # Count number of colors used
    num_colors = max(color_map.values()) + 1

    # Create a target number of colors (1 less than current)
    target_colors = num_colors - 1

    # Initialize tabu list
    tabu_list = {}  # Maps (vertex, color) to iteration when it can be used again
    iteration = 0

    while iteration < max_iterations and target_colors >= 1:
        # Try to reduce the number of colors to target_colors
        temp_color_map = color_map.copy()

        # Reassign vertices with the highest color to other colors
        for vertex in [v for v, c in temp_color_map.items() if c == target_colors]:
            # Try each color less than target
            legal_colors = []
            for color in range(target_colors):
                # Check if the move is legal (no conflicts with neighbors)
                if all(temp_color_map.get(neighbor) != color for neighbor in graph.get_neighbors(vertex)):
                    legal_colors.append(color)

            # If no legal color found, choose one that minimizes conflicts
            if not legal_colors:
                # Count conflicts for each color
                conflicts = [sum(1 for neighbor in graph.get_neighbors(vertex)
                                 if temp_color_map.get(neighbor) == color)
                             for color in range(target_colors)]

                # Find colors with minimum conflicts, excluding tabu moves if possible
                min_conflicts = min(conflicts)
                candidates = [color for color, conf in enumerate(conflicts) if conf == min_conflicts]

                # Exclude tabu moves unless they lead to a better solution
                non_tabu = [color for color in candidates
                            if (vertex, color) not in tabu_list or tabu_list[(vertex, color)] <= iteration]

                if non_tabu:
                    selected_color = random.choice(non_tabu)
                else:
                    # Aspiration criterion: accept tabu move if it's the best so far
                    selected_color = random.choice(candidates)
            else:
                # Choose a legal color, prefering non-tabu moves
                non_tabu = [color for color in legal_colors
                            if (vertex, color) not in tabu_list or tabu_list[(vertex, color)] <= iteration]

                if non_tabu:
                    selected_color = random.choice(non_tabu)
                else:
                    selected_color = random.choice(legal_colors)

            # Make the move
            old_color = temp_color_map[vertex]
            temp_color_map[vertex] = selected_color

            # Add to tabu list
            tabu_list[(vertex, old_color)] = iteration + tabu_tenure

        # Check if the solution is valid (no conflicts)
        valid = True
        for vertex in graph.vertices:
            for neighbor in graph.get_neighbors(vertex):
                if temp_color_map[vertex] == temp_color_map[neighbor]:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            # Update solution and reduce target
            color_map = temp_color_map
            num_colors = target_colors
            target_colors -= 1

        iteration += 1

    return color_map


def is_valid_coloring(graph, coloring: Dict[Any, int]) -> bool:
    """
    Check if a coloring is valid (no adjacent vertices have the same color).

    Args:
        graph: The graph to check
        coloring: Dictionary mapping vertices to colors

    Returns:
        True if the coloring is valid, False otherwise
    """
    # Check each edge
    for u, v in graph.edges:
        if u in coloring and v in coloring and coloring[u] == coloring[v]:
            return False

    return True


def count_colors(coloring: Dict[Any, int]) -> int:
    """
    Count the number of colors used in a coloring.

    Args:
        coloring: Dictionary mapping vertices to colors

    Returns:
        Number of distinct colors used
    """
    return len(set(coloring.values()))


def kempe_chain_interchange(graph, coloring: Dict[Any, int], vertex: Any, new_color: int) -> Dict[Any, int]:
    """
    Perform a Kempe chain interchange to recolor a vertex.

    This operation preserves the validity of a coloring while changing the
    color of a specific vertex.

    Args:
        graph: The graph
        coloring: Current valid coloring
        vertex: Vertex to recolor
        new_color: New color for the vertex

    Returns:
        Updated coloring
    """
    if vertex not in coloring:
        raise ValueError("Vertex not in coloring")

    old_color = coloring[vertex]

    if old_color == new_color:
        return coloring.copy()

    # Find the Kempe chain - the connected component containing the vertex
    # in the subgraph induced by vertices colored with old_color or new_color
    chain = set()
    visited = set()
    queue = collections.deque([vertex])
    visited.add(vertex)

    while queue:
        u = queue.popleft()
        chain.add(u)

        for v in graph.get_neighbors(u):
            if v in coloring and coloring[v] in (old_color, new_color) and v not in visited:
                visited.add(v)
                queue.append(v)

    # Flip the colors in the chain
    new_coloring = coloring.copy()
    for u in chain:
        if new_coloring[u] == old_color:
            new_coloring[u] = new_color
        else:
            new_coloring[u] = old_color

    return new_coloring