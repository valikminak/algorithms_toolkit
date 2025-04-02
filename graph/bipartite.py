from typing import List, Dict, Tuple, Set, Optional, Any, Union, Callable
import collections


def is_bipartite(graph) -> Tuple[bool, Set[Any], Set[Any]]:
    """
    Check if a graph is bipartite and return the two vertex sets.

    A graph is bipartite if its vertices can be divided into two disjoint sets
    such that every edge connects vertices from different sets.

    Args:
        graph: The graph to check

    Returns:
        Tuple of (is_bipartite, left_set, right_set) where:
        - is_bipartite: True if the graph is bipartite, False otherwise
        - left_set: Set of vertices in the first partition
        - right_set: Set of vertices in the second partition
    """
    if not graph.vertices:
        return True, set(), set()

    color = {}  # 0 for left set, 1 for right set

    for start_vertex in graph.vertices:
        if start_vertex not in color:
            queue = collections.deque([start_vertex])
            color[start_vertex] = 0

            while queue:
                u = queue.popleft()

                for v in graph.get_neighbors(u):
                    if v not in color:
                        color[v] = 1 - color[u]  # Assign opposite color
                        queue.append(v)
                    elif color[v] == color[u]:
                        # If adjacent vertices have the same color, graph is not bipartite
                        return False, set(), set()

    left_set = {v for v, c in color.items() if c == 0}
    right_set = {v for v, c in color.items() if c == 1}

    return True, left_set, right_set


def maximum_bipartite_matching(graph) -> Tuple[Dict[Any, Any], int]:
    """
    Find a maximum bipartite matching using the Ford-Fulkerson algorithm.

    A matching in a bipartite graph is a set of edges with no common vertices.

    Args:
        graph: The bipartite graph

    Returns:
        Tuple of (matching, size) where:
        - matching: Dictionary mapping vertices in the left set to matched vertices in the right set
        - size: Size of the maximum matching
    """
    # Check if graph is bipartite
    is_bip, left_set, right_set = is_bipartite(graph)

    if not is_bip:
        raise ValueError("Input graph is not bipartite")

    # Use Ford-Fulkerson algorithm to find maximum matching
    matching = {}

    # Try to find an augmenting path for each unmatched vertex in left set
    while True:
        # Find an augmenting path using BFS
        path = _find_augmenting_path(graph, matching, left_set, right_set)

        if not path:
            break

        # Augment the matching along the path
        for i in range(0, len(path) - 1, 2):
            u, v = path[i], path[i + 1]
            matching[u] = v

    return matching, len(matching)


def _find_augmenting_path(graph, matching, left_set, right_set):
    """
    Find an augmenting path in a bipartite graph.

    An augmenting path starts at an unmatched vertex in the left set, alternates
    between unmatched and matched edges, and ends at an unmatched vertex in the right set.

    Args:
        graph: The bipartite graph
        matching: Current matching as a dictionary
        left_set: Set of vertices in the left partition
        right_set: Set of vertices in the right partition

    Returns:
        List of vertices forming an augmenting path, or empty list if none exists
    """
    # Create an inverse mapping for matched vertices in the right set
    inv_matching = {v: u for u, v in matching.items()}

    # Start from unmatched vertices in the left set
    unmatched_left = left_set - set(matching.keys())

    # BFS to find an augmenting path
    visited = set()
    paths = {}  # Maps vertices to the path that reached them

    queue = collections.deque()
    for u in unmatched_left:
        queue.append(u)
        paths[u] = [u]
        visited.add(u)

    # BFS until we reach an unmatched vertex in the right set
    while queue:
        u = queue.popleft()

        if u in left_set:
            # If u is in the left set, follow unmatched edges
            for v in graph.get_neighbors(u):
                if v not in visited:
                    # Check if v is unmatched in the right set
                    if v not in inv_matching:
                        # Found an augmenting path
                        return paths[u] + [v]

                    # Add v and its match to the path
                    visited.add(v)
                    paths[v] = paths[u] + [v]
                    queue.append(v)
        else:
            # If u is in the right set, follow a matched edge
            if u in inv_matching:
                v = inv_matching[u]
                if v not in visited:
                    visited.add(v)
                    paths[v] = paths[u] + [v]
                    queue.append(v)

    return []  # No augmenting path found


def hopcroft_karp(graph) -> Tuple[Dict[Any, Any], int]:
    """
    Find a maximum bipartite matching using the Hopcroft-Karp algorithm.

    This algorithm is more efficient than the simple Ford-Fulkerson implementation
    for bipartite matching, with a complexity of O(E√V).

    Args:
        graph: The bipartite graph

    Returns:
        Tuple of (matching, size) where:
        - matching: Dictionary mapping vertices in the left set to matched vertices in the right set
        - size: Size of the maximum matching
    """
    # Check if graph is bipartite
    is_bip, left_set, right_set = is_bipartite(graph)

    if not is_bip:
        raise ValueError("Input graph is not bipartite")

    # Initialize matching
    matching = {}
    inv_matching = {}  # Right to left mapping

    # Find augmenting paths in batches using BFS
    while True:
        # Find multiple disjoint augmenting paths using BFS
        paths = _find_augmenting_paths_bfs(graph, matching, inv_matching, left_set, right_set)

        if not paths:
            break

        # Augment the matching along each path
        for path in paths:
            for i in range(0, len(path) - 1, 2):
                u, v = path[i], path[i + 1]
                matching[u] = v
                inv_matching[v] = u

    return matching, len(matching)


def _find_augmenting_paths_bfs(graph, matching, inv_matching, left_set, right_set):
    """
    Find multiple disjoint augmenting paths using BFS.

    Args:
        graph: The bipartite graph
        matching: Current matching (left to right)
        inv_matching: Inverse of current matching (right to left)
        left_set: Set of vertices in the left partition
        right_set: Set of vertices in the right partition

    Returns:
        List of augmenting paths
    """
    NIL = object()  # Sentinel for unmatched vertices

    # Set of unmatched vertices in the left set
    unmatched_left = left_set - set(matching.keys())

    # Distances for BFS
    dist = {}
    for u in left_set:
        dist[u] = float('inf') if u in matching else 0
    dist[NIL] = float('inf')

    # BFS to find shortest augmenting paths
    queue = collections.deque()
    for u in unmatched_left:
        queue.append(u)

    while queue:
        u = queue.popleft()

        # Skip if u has no hope of being part of a shorter augmenting path
        if dist[u] < dist[NIL]:
            for v in graph.get_neighbors(u):
                # Get the vertex matched to v, or NIL if unmatched
                match_v = inv_matching.get(v, NIL)

                # Only consider if it leads to a shorter augmenting path
                if dist[match_v] == float('inf'):
                    dist[match_v] = dist[u] + 1
                    queue.append(match_v)

    # If no augmenting path was found
    if dist[NIL] == float('inf'):
        return []

    # Use DFS to find multiple disjoint augmenting paths
    paths = []
    visited = set()

    for u in unmatched_left:
        if _dfs_for_augmenting_path(graph, matching, inv_matching, dist, u, visited, NIL):
            # Reconstruct the path
            path = [u]
            current = u
            while current in matching:
                v = matching[current]
                path.append(v)
                current = inv_matching.get(v)
                if current:
                    path.append(current)

            paths.append(path)

    return paths


def _dfs_for_augmenting_path(graph, matching, inv_matching, dist, u, visited, NIL):
    """
    DFS to find an augmenting path starting at u.

    Args:
        graph: The bipartite graph
        matching: Current matching (left to right)
        inv_matching: Inverse of current matching (right to left)
        dist: Distance labels from BFS
        u: Current vertex
        visited: Set of visited vertices
        NIL: Sentinel for unmatched vertices

    Returns:
        True if an augmenting path was found, False otherwise
    """
    visited.add(u)

    for v in graph.get_neighbors(u):
        match_v = inv_matching.get(v, NIL)

        # Check if this edge can be part of an augmenting path
        if match_v not in visited and dist[match_v] == dist[u] + 1:
            if match_v == NIL or _dfs_for_augmenting_path(graph, matching, inv_matching, dist, match_v, visited, NIL):
                # Augment the matching
                matching[u] = v
                inv_matching[v] = u
                return True

    return False


def minimum_vertex_cover_bipartite(graph) -> Set[Any]:
    """
    Find a minimum vertex cover in a bipartite graph.

    A vertex cover is a set of vertices such that each edge of the graph
    is incident to at least one vertex in the set.

    König's theorem states that in a bipartite graph, the size of a maximum matching
    equals the size of a minimum vertex cover.

    Args:
        graph: The bipartite graph

    Returns:
        Set of vertices forming a minimum vertex cover
    """
    # Check if graph is bipartite
    is_bip, left_set, right_set = is_bipartite(graph)

    if not is_bip:
        raise ValueError("Input graph is not bipartite")

    # Find maximum matching
    matching, _ = maximum_bipartite_matching(graph)

    # Inverse matching (right to left)
    inv_matching = {v: u for u, v in matching.items()}

    # Find unmatched vertices in the left set
    unmatched_left = left_set - set(matching.keys())

    # Find vertices reachable from unmatched left vertices in the alternating graph
    reachable = set()

    # BFS from unmatched left vertices
    queue = collections.deque(unmatched_left)
    for u in unmatched_left:
        reachable.add(u)

    while queue:
        u = queue.popleft()

        if u in left_set:
            # From left vertex, follow any edge
            for v in graph.get_neighbors(u):
                if v not in reachable:
                    reachable.add(v)
                    queue.append(v)
        else:
            # From right vertex, only follow matched edges
            if u in inv_matching:
                v = inv_matching[u]
                if v not in reachable:
                    reachable.add(v)
                    queue.append(v)

    # Minimum vertex cover = (Left - Reachable) ∪ (Right ∩ Reachable)
    cover = (left_set - reachable) | (right_set & reachable)

    return cover


def maximum_independent_set_bipartite(graph) -> Set[Any]:
    """
    Find a maximum independent set in a bipartite graph.

    An independent set is a set of vertices such that no two vertices in the set
    are adjacent. In a bipartite graph, the complement of a minimum vertex cover
    is a maximum independent set.

    Args:
        graph: The bipartite graph

    Returns:
        Set of vertices forming a maximum independent set
    """
    # Find minimum vertex cover
    cover = minimum_vertex_cover_bipartite(graph)

    # Maximum independent set is the complement of the cover
    return set(graph.vertices) - cover