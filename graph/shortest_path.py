import heapq
from typing import List, Dict, Tuple, Optional, Any, Callable
import math


def dijkstra(graph, start: Any) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]]]:
    """
    Dijkstra's algorithm for finding shortest paths in a weighted graph.

    Args:
        graph: The weighted graph
        start: The starting vertex

    Returns:
        Tuple of (distances, predecessors) where:
        - distances: Dictionary mapping each vertex to its shortest distance from start
        - predecessors: Dictionary mapping each vertex to its predecessor in the shortest path
    """
    if not graph.weighted:
        raise ValueError("Dijkstra's algorithm requires a weighted graph")

    if start not in graph.vertices:
        return {}, {}

    # Initialize
    distances = {vertex: float('infinity') for vertex in graph.vertices}
    distances[start] = 0
    predecessors = {vertex: None for vertex in graph.vertices}

    # Priority queue of (distance, vertex)
    pq = [(0, start)]
    visited = set()

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        # If we've already found a shorter path, skip
        if current_distance > distances[current_vertex]:
            continue

        # Mark as visited
        visited.add(current_vertex)

        # Check all neighbors
        for neighbor in graph.get_neighbors(current_vertex):
            if neighbor in visited:
                continue

            weight = graph.get_edge_weight(current_vertex, neighbor)
            distance = current_distance + weight

            # If we found a shorter path, update
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, predecessors


def bellman_ford(graph, start: Any) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]], bool]:
    """
    Bellman-Ford algorithm for finding shortest paths in a weighted graph,
    including graphs with negative edge weights (but no negative cycles).

    Args:
        graph: The weighted graph
        start: The starting vertex

    Returns:
        Tuple of (distances, predecessors, no_negative_cycle) where:
        - distances: Dictionary mapping each vertex to its shortest distance from start
        - predecessors: Dictionary mapping each vertex to its predecessor in the shortest path
        - no_negative_cycle: True if no negative cycle was detected, False otherwise
    """
    if not graph.weighted:
        raise ValueError("Bellman-Ford algorithm requires a weighted graph")

    if start not in graph.vertices:
        return {}, {}, True

    # Initialize
    distances = {vertex: float('infinity') for vertex in graph.vertices}
    distances[start] = 0
    predecessors = {vertex: None for vertex in graph.vertices}

    # Relax edges |V| - 1 times
    for _ in range(len(graph.vertices) - 1):
        for u, v, weight in graph.get_edges():
            if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u

    # Check for negative cycles
    for u, v, weight in graph.get_edges():
        if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
            return distances, predecessors, False  # Negative cycle detected

    return distances, predecessors, True


def floyd_warshall(graph) -> Tuple[Dict[Tuple[Any, Any], float], Dict[Tuple[Any, Any], Optional[Any]]]:
    """
    Floyd-Warshall algorithm for finding all-pairs shortest paths in a weighted graph.

    Args:
        graph: The weighted graph

    Returns:
        Tuple of (distances, next_vertices) where:
        - distances: Dictionary mapping each vertex pair (u, v) to the shortest distance from u to v
        - next_vertices: Dictionary for path reconstruction, mapping each pair (u, v) to the next vertex in the path
    """
    if not graph.weighted:
        raise ValueError("Floyd-Warshall algorithm requires a weighted graph")

    # Initialize distances and next vertices
    vertices = list(graph.vertices)
    distances = {}
    next_vertices = {}

    # Set initial distances
    for u in vertices:
        for v in vertices:
            if u == v:
                distances[(u, v)] = 0
                next_vertices[(u, v)] = None
            elif (u, v) in graph.edges:
                distances[(u, v)] = graph.get_edge_weight(u, v)
                next_vertices[(u, v)] = v
            else:
                distances[(u, v)] = float('infinity')
                next_vertices[(u, v)] = None

    # Floyd-Warshall algorithm
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if distances[(i, k)] != float('infinity') and distances[(k, j)] != float('infinity'):
                    if distances[(i, j)] > distances[(i, k)] + distances[(k, j)]:
                        distances[(i, j)] = distances[(i, k)] + distances[(k, j)]
                        next_vertices[(i, j)] = next_vertices[(i, k)]

    return distances, next_vertices


def a_star_search(graph, start: Any, goal: Any, heuristic: Callable[[Any, Any], float]) -> Tuple[
    Dict[Any, float], Dict[Any, Optional[Any]]]:
    """
    A* Search algorithm for finding the shortest path from start to goal using a heuristic.

    Args:
        graph: The weighted graph
        start: The starting vertex
        goal: The goal vertex
        heuristic: A function that takes (current, goal) and returns an estimated distance

    Returns:
        Tuple of (distances, predecessors) where:
        - distances: Dictionary mapping each visited vertex to its shortest distance from start
        - predecessors: Dictionary mapping each visited vertex to its predecessor in the shortest path
    """
    if not graph.weighted:
        raise ValueError("A* Search requires a weighted graph")

    if start not in graph.vertices or goal not in graph.vertices:
        return {}, {}

    # Priority queue: (f_score, vertex)
    # f_score = g_score + heuristic, where g_score is the cost from start to current vertex
    open_set = [(heuristic(start, goal), start)]
    closed_set = set()

    # g_score: cost from start to current node
    g_score = {vertex: float('infinity') for vertex in graph.vertices}
    g_score[start] = 0

    # f_score: estimated total cost from start to goal through current node
    f_score = {vertex: float('infinity') for vertex in graph.vertices}
    f_score[start] = heuristic(start, goal)

    # For path reconstruction
    predecessors = {vertex: None for vertex in graph.vertices}

    while open_set:
        _, current = heapq.heappop(open_set)

        # If we reached the goal, we're done
        if current == goal:
            distances = {v: g_score[v] for v in g_score if g_score[v] != float('infinity')}
            return distances, predecessors

        closed_set.add(current)

        # Check all neighbors
        for neighbor in graph.get_neighbors(current):
            if neighbor in closed_set:
                continue

            # Calculate tentative g_score via current
            edge_weight = graph.get_edge_weight(current, neighbor)
            tentative_g_score = g_score[current] + edge_weight

            # If we found a better path
            if tentative_g_score < g_score[neighbor]:
                # Update path
                predecessors[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)

                # Add to open set with updated priority
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # No path found
    distances = {v: g_score[v] for v in g_score if g_score[v] != float('infinity')}
    return distances, predecessors


# graph/shortest_path.py (continued)

def bidirectional_dijkstra(graph, start: Any, goal: Any) -> Tuple[float, List[Any]]:
    """
    Bidirectional Dijkstra's algorithm for finding shortest path between start and goal.

    Args:
        graph: The weighted graph
        start: The starting vertex
        goal: The goal vertex

    Returns:
        Tuple of (distance, path) where:
        - distance: The shortest distance from start to goal
        - path: List of vertices forming the shortest path
    """
    if not graph.weighted:
        raise ValueError("Bidirectional Dijkstra requires a weighted graph")

    if start not in graph.vertices or goal not in graph.vertices:
        return float('infinity'), []

    if start == goal:
        return 0, [start]

    # Forward search from start
    forward_visited = set()
    forward_distances = {vertex: float('infinity') for vertex in graph.vertices}
    forward_distances[start] = 0
    forward_predecessors = {vertex: None for vertex in graph.vertices}
    forward_queue = [(0, start)]

    # Backward search from goal
    backward_visited = set()
    backward_distances = {vertex: float('infinity') for vertex in graph.vertices}
    backward_distances[goal] = 0
    backward_predecessors = {vertex: None for vertex in graph.vertices}
    backward_queue = [(0, goal)]

    # Best distance found so far
    best_distance = float('infinity')
    meeting_point = None

    # Alternating between forward and backward search
    while forward_queue and backward_queue:
        # Forward step
        _, current_forward = heapq.heappop(forward_queue)

        # If already visited in backward search, check if this gives a better path
        if current_forward in backward_visited:
            distance = forward_distances[current_forward] + backward_distances[current_forward]
            if distance < best_distance:
                best_distance = distance
                meeting_point = current_forward

        # If we can't improve the best distance found, stop
        if forward_distances[current_forward] > best_distance:
            break

        forward_visited.add(current_forward)

        # Process neighbors in forward direction
        for neighbor in graph.get_neighbors(current_forward):
            if neighbor in forward_visited:
                continue

            weight = graph.get_edge_weight(current_forward, neighbor)
            distance = forward_distances[current_forward] + weight

            if distance < forward_distances[neighbor]:
                forward_distances[neighbor] = distance
                forward_predecessors[neighbor] = current_forward
                heapq.heappush(forward_queue, (distance, neighbor))

                # Check if this improves the best path
                if neighbor in backward_visited:
                    total_distance = distance + backward_distances[neighbor]
                    if total_distance < best_distance:
                        best_distance = total_distance
                        meeting_point = neighbor

        # Backward step
        _, current_backward = heapq.heappop(backward_queue)

        # If already visited in forward search, check if this gives a better path
        if current_backward in forward_visited:
            distance = forward_distances[current_backward] + backward_distances[current_backward]
            if distance < best_distance:
                best_distance = distance
                meeting_point = current_backward

        # If we can't improve the best distance found, stop
        if backward_distances[current_backward] > best_distance:
            break

        backward_visited.add(current_backward)

        # Process neighbors in backward direction
        for neighbor in graph.get_neighbors(current_backward):
            if neighbor in backward_visited:
                continue

            weight = graph.get_edge_weight(current_backward, neighbor)
            distance = backward_distances[current_backward] + weight

            if distance < backward_distances[neighbor]:
                backward_distances[neighbor] = distance
                backward_predecessors[neighbor] = current_backward
                heapq.heappush(backward_queue, (distance, neighbor))

                # Check if this improves the best path
                if neighbor in forward_visited:
                    total_distance = forward_distances[neighbor] + distance
                    if total_distance < best_distance:
                        best_distance = total_distance
                        meeting_point = neighbor

    # Reconstruct the path
    if meeting_point is None:
        return float('infinity'), []  # No path found

    # Forward path: start -> meeting_point
    forward_path = []
    current = meeting_point
    while current is not None:
        forward_path.append(current)
        current = forward_predecessors[current]
    forward_path.reverse()

    # Backward path: meeting_point -> goal
    backward_path = []
    current = backward_predecessors[meeting_point]
    while current is not None:
        backward_path.append(current)
        current = backward_predecessors[current]
    backward_path.reverse()

    # Complete path: start -> meeting_point -> goal
    path = forward_path + backward_path

    return best_distance, path


def reconstruct_shortest_path(start: Any, end: Any, predecessors: Dict[Any, Optional[Any]]) -> List[Any]:
    """
    Reconstruct shortest path from predecessors dictionary.

    Args:
        start: Starting vertex
        end: Ending vertex
        predecessors: Dictionary mapping each vertex to its predecessor

    Returns:
        List of vertices forming the shortest path from start to end.
        Returns empty list if no path exists.
    """
    if end not in predecessors or predecessors[end] is None and start != end:
        return []  # No path exists

    path = []
    current = end

    while current is not None:
        path.append(current)
        current = predecessors[current]

    return list(reversed(path))


def greedy_best_first_search(graph, start: Any, goal: Any, heuristic: Callable[[Any, Any], float]) -> List[Any]:
    """
    Greedy Best-First Search algorithm using a heuristic function.

    This algorithm is like A* but uses only the heuristic for evaluation,
    not considering the path cost.

    Args:
        graph: The graph
        start: The starting vertex
        goal: The goal vertex
        heuristic: A function that takes (current, goal) and returns an estimated distance

    Returns:
        List of vertices forming a path from start to goal, or empty list if no path exists
    """
    if start not in graph.vertices or goal not in graph.vertices:
        return []

    # Open set with priority based only on heuristic
    open_set = [(heuristic(start, goal), start)]
    closed_set = set()

    # For path reconstruction
    predecessors = {start: None}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = predecessors.get(current)
            return list(reversed(path))

        closed_set.add(current)

        for neighbor in graph.get_neighbors(current):
            if neighbor in closed_set:
                continue

            if neighbor not in predecessors:
                predecessors[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor))

    return []  # No path found


# Common heuristic functions for A* and Greedy Best-First Search

def manhattan_distance(node1, node2):
    """
    Calculate Manhattan distance between two points.

    Args:
        node1: First point as (x, y) tuple or object with x, y attributes
        node2: Second point as (x, y) tuple or object with x, y attributes

    Returns:
        Manhattan distance between the points
    """
    # Get coordinates from tuples or objects
    try:
        x1, y1 = node1
        x2, y2 = node2
    except (TypeError, ValueError):
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y

    return abs(x1 - x2) + abs(y1 - y2)


def euclidean_distance(node1, node2):
    """
    Calculate Euclidean distance between two points.

    Args:
        node1: First point as (x, y) tuple or object with x, y attributes
        node2: Second point as (x, y) tuple or object with x, y attributes

    Returns:
        Euclidean distance between the points
    """
    # Get coordinates from tuples or objects
    try:
        x1, y1 = node1
        x2, y2 = node2
    except (TypeError, ValueError):
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def chebyshev_distance(node1, node2):
    """
    Calculate Chebyshev distance (maximum of the absolute differences).

    Args:
        node1: First point as (x, y) tuple or object with x, y attributes
        node2: Second point as (x, y) tuple or object with x, y attributes

    Returns:
        Chebyshev distance between the points
    """
    # Get coordinates from tuples or objects
    try:
        x1, y1 = node1
        x2, y2 = node2
    except (TypeError, ValueError):
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y

    return max(abs(x1 - x2), abs(y1 - y2))


def custom_heuristic(node, goal, graph=None, weight_func=None):
    """
    Create a custom heuristic that can use graph information.

    Args:
        node: Current node
        goal: Goal node
        graph: The graph (optional)
        weight_func: Function to calculate edge cost (optional)

    Returns:
        A heuristic value
    """
    # Default to Euclidean distance if no custom logic provided
    return euclidean_distance(node, goal)
