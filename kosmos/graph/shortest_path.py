from typing import List, Dict, Tuple, Optional, Any, Callable
import math
import heapq


def dijkstra(graph, start_vertex, end_vertex=None):
    """
    Dijkstra's shortest path algorithm.

    Args:
        graph: The weighted graph
        start_vertex: Starting vertex
        end_vertex: Optional end vertex

    Returns:
        Tuple of (distances, predecessors)
    """
    if not graph.weighted:
        raise ValueError("Dijkstra's algorithm requires a weighted graph")

    # Initialize distances and predecessors
    distances = {vertex: float('infinity') for vertex in graph.vertices}
    distances[start_vertex] = 0
    predecessors = {vertex: None for vertex in graph.vertices}

    # Priority queue
    pq = [(0, start_vertex)]

    # Set of visited vertices
    visited = set()

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        # If we've visited this vertex before with a better path, skip it
        if current_vertex in visited:
            continue

        # Mark as visited
        visited.add(current_vertex)

        # If we've reached the end vertex, we can stop
        if end_vertex and current_vertex == end_vertex:
            break

        # Check all neighbors
        for neighbor, weight in graph.get_neighbors_with_weights(current_vertex).items():
            # Calculate new distance
            distance = current_distance + weight

            # If we found a better path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, predecessors


def bellman_ford(graph, start_vertex):
    """
    Bellman-Ford shortest path algorithm.
    Works with negative edge weights and can detect negative cycles.

    Args:
        graph: The weighted graph
        start_vertex: Starting vertex

    Returns:
        Tuple of (distances, predecessors) or False if negative cycle exists
    """
    if not graph.weighted:
        raise ValueError("Bellman-Ford algorithm requires a weighted graph")

    # Initialize distances and predecessors
    distances = {vertex: float('infinity') for vertex in graph.vertices}
    distances[start_vertex] = 0
    predecessors = {vertex: None for vertex in graph.vertices}

    # Relax edges |V| - 1 times
    for _ in range(len(graph.vertices) - 1):
        for edge, weight in graph.edges.items():
            source, target = edge.split(',')

            # Skip if the source vertex is unreachable
            if distances[source] == float('infinity'):
                continue

            # Relax the edge
            if distances[source] + weight < distances[target]:
                distances[target] = distances[source] + weight
                predecessors[target] = source

    # Check for negative-weight cycles
    for edge, weight in graph.edges.items():
        source, target = edge.split(',')

        if distances[source] != float('infinity') and distances[source] + weight < distances[target]:
            return False  # Negative cycle detected

    return distances, predecessors


def floyd_warshall(graph):
    """
    Floyd-Warshall algorithm for all-pairs shortest paths.

    Args:
        graph: The weighted graph

    Returns:
        Dictionary of dictionaries representing shortest path distances
    """
    if not graph.weighted:
        raise ValueError("Floyd-Warshall algorithm requires a weighted graph")

    # Initialize distance matrix
    dist = {}
    for u in graph.vertices:
        dist[u] = {}
        for v in graph.vertices:
            if u == v:
                dist[u][v] = 0
            else:
                dist[u][v] = graph.get_edge_weight(u, v) or float('infinity')

    # Main algorithm
    for k in graph.vertices:
        for i in graph.vertices:
            for j in graph.vertices:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


def a_star(graph, start_vertex, end_vertex, heuristic):
    """
    A* search algorithm for finding shortest path.

    Args:
        graph: The weighted graph
        start_vertex: Starting vertex
        end_vertex: Target vertex
        heuristic: Function that estimates distance from vertex to end

    Returns:
        Tuple of (path, cost) or ([], infinity) if no path exists
    """
    if not graph.weighted:
        raise ValueError("A* algorithm requires a weighted graph")

    # Open set: (f_score, g_score, vertex, path)
    open_set = [(heuristic(start_vertex, end_vertex), 0, start_vertex, [])]
    heapq.heapify(open_set)

    # Closed set
    closed_set = set()

    # Track best g_scores found so far
    g_scores = {vertex: float('infinity') for vertex in graph.vertices}
    g_scores[start_vertex] = 0

    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)

        # Skip if we've already found a better path to this vertex
        if g_score > g_scores[current]:
            continue

        # If we've reached the target
        if current == end_vertex:
            return path + [current], g_score

        # Skip if we've already visited this vertex
        if current in closed_set:
            continue

        # Mark as visited
        closed_set.add(current)

        # Process neighbors
        for neighbor, weight in graph.get_neighbors_with_weights(current).items():
            # Skip if already visited
            if neighbor in closed_set:
                continue

            # Calculate tentative g_score
            tentative_g = g_score + weight

            # Skip if we've already found a better path to this neighbor
            if tentative_g >= g_scores[neighbor]:
                continue

            # Update best path
            g_scores[neighbor] = tentative_g

            # Calculate f_score (g + heuristic)
            f = tentative_g + heuristic(neighbor, end_vertex)

            # Add to open set
            heapq.heappush(open_set, (f, tentative_g, neighbor, path + [current]))

    # No path found
    return [], float('infinity')


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
