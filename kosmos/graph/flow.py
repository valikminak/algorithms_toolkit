from typing import List, Dict, Tuple, Set, Any
import collections


def ford_fulkerson(graph, source: Any, sink: Any) -> Tuple[float, Dict[Tuple[Any, Any], float]]:
    """
    Ford-Fulkerson algorithm for finding the maximum flow in a flow network.

    Args:
        graph: The flow network graph
        source: The source vertex
        sink: The sink vertex

    Returns:
        Tuple of (max_flow, flow_dict) where:
        - max_flow: The maximum flow value from source to sink
        - flow_dict: Dictionary mapping each edge (u, v) to its flow value
    """
    if not graph.weighted:
        raise ValueError("Ford-Fulkerson requires a weighted graph")

    if source not in graph.vertices or sink not in graph.vertices:
        return 0, {}

    # Initialize residual graph and flow
    residual_graph = _create_residual_graph(graph)
    flow = {(u, v): 0 for u, v in residual_graph}

    # Find augmenting paths and update flow
    max_flow = 0
    while True:
        # Find an augmenting path using BFS
        path, bottleneck = _find_augmenting_path_bfs(residual_graph, source, sink)

        # If no path exists, we're done
        if path is None:
            break

        # Update the flow along the path
        max_flow += bottleneck
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            # Forward edge: increase flow
            flow[(u, v)] += bottleneck
            residual_graph[(u, v)] -= bottleneck

            # Backward edge: decrease flow (increase residual capacity)
            residual_graph[(v, u)] += bottleneck

    # Filter out zero flows and backward edges
    flow_dict = {(u, v): f for (u, v), f in flow.items() if f > 0 and (u, v) in graph.edges}

    return max_flow, flow_dict


def edmonds_karp(graph, source: Any, sink: Any) -> Tuple[float, Dict[Tuple[Any, Any], float]]:
    """
    Edmonds-Karp algorithm for finding the maximum flow in a flow network.

    This is a specific implementation of Ford-Fulkerson that always uses BFS
    to find augmenting paths.

    Args:
        graph: The flow network graph
        source: The source vertex
        sink: The sink vertex

    Returns:
        Tuple of (max_flow, flow_dict) where:
        - max_flow: The maximum flow value from source to sink
        - flow_dict: Dictionary mapping each edge (u, v) to its flow value
    """
    # Edmonds-Karp is essentially Ford-Fulkerson with BFS, which we already use
    return ford_fulkerson(graph, source, sink)


def dinic(graph, source: Any, sink: Any) -> Tuple[float, Dict[Tuple[Any, Any], float]]:
    """
    Dinic's algorithm for finding the maximum flow in a flow network.

    This algorithm uses level graphs and blocking flows for better performance
    than Ford-Fulkerson or Edmonds-Karp.

    Args:
        graph: The flow network graph
        source: The source vertex
        sink: The sink vertex

    Returns:
        Tuple of (max_flow, flow_dict) where:
        - max_flow: The maximum flow value from source to sink
        - flow_dict: Dictionary mapping each edge (u, v) to its flow value
    """
    if not graph.weighted:
        raise ValueError("Dinic's algorithm requires a weighted graph")

    if source not in graph.vertices or sink not in graph.vertices:
        return 0, {}

    # Initialize residual graph and flow
    residual_graph = _create_residual_graph(graph)
    flow = {(u, v): 0 for u, v in residual_graph}

    max_flow = 0

    while True:
        # Create level graph using BFS
        level = _create_level_graph(residual_graph, source, sink)

        # If sink is not reachable, we're done
        if level[sink] == -1:
            break

        # Find blocking flow in the level graph
        while True:
            # Use DFS to find an augmenting path in the level graph
            path, bottleneck = _find_augmenting_path_dfs(residual_graph, source, sink, level)

            # If no path exists, this level graph is exhausted
            if path is None:
                break

            # Update the flow along the path
            max_flow += bottleneck
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                # Forward edge: increase flow
                flow[(u, v)] += bottleneck
                residual_graph[(u, v)] -= bottleneck

                # Backward edge: decrease flow (increase residual capacity)
                residual_graph[(v, u)] += bottleneck

    # Filter out zero flows and backward edges
    flow_dict = {(u, v): f for (u, v), f in flow.items() if f > 0 and (u, v) in graph.edges}

    return max_flow, flow_dict


def push_relabel(graph, source: Any, sink: Any) -> Tuple[float, Dict[Tuple[Any, Any], float]]:
    """
    Push-Relabel algorithm for finding the maximum flow in a flow network.

    This algorithm uses a different approach than augmenting path methods,
    and has better theoretical time complexity for dense graphs.

    Args:
        graph: The flow network graph
        source: The source vertex
        sink: The sink vertex

    Returns:
        Tuple of (max_flow, flow_dict) where:
        - max_flow: The maximum flow value from source to sink
        - flow_dict: Dictionary mapping each edge (u, v) to its flow value
    """
    if not graph.weighted:
        raise ValueError("Push-Relabel requires a weighted graph")

    if source not in graph.vertices or sink not in graph.vertices:
        return 0, {}

    # Create residual graph
    residual_graph = _create_residual_graph(graph)
    flow = {(u, v): 0 for u, v in residual_graph}

    # Initialize heights and excess flow
    height = {v: 0 for v in graph.vertices}
    excess = {v: 0 for v in graph.vertices}

    # Set height of source to n (number of vertices)
    height[source] = len(graph.vertices)

    # Preflow: push as much flow as possible from source to its neighbors
    for v in graph.get_neighbors(source):
        capacity = graph.get_edge_weight(source, v)
        flow[(source, v)] = capacity
        flow[(v, source)] = -capacity  # Reverse flow for residual graph
        excess[v] = capacity
        excess[source] -= capacity
        residual_graph[(source, v)] -= capacity
        residual_graph[(v, source)] += capacity

    # Process vertices with excess flow
    active = [v for v in graph.vertices if v != source and v != sink and excess[v] > 0]

    while active:
        v = active[0]

        # Try to push excess flow to neighbors
        pushed = False
        for neighbor in graph.get_neighbors(v):
            # Can only push if residual capacity > 0 and height difference is correct
            if residual_graph[(v, neighbor)] > 0 and height[v] == height[neighbor] + 1:
                # Push flow from v to neighbor
                push_amount = min(excess[v], residual_graph[(v, neighbor)])
                flow[(v, neighbor)] += push_amount
                flow[(neighbor, v)] -= push_amount
                residual_graph[(v, neighbor)] -= push_amount
                residual_graph[(neighbor, v)] += push_amount
                excess[v] -= push_amount
                excess[neighbor] += push_amount

                # Add neighbor to active list if it has excess flow now and isn't source/sink
                if excess[neighbor] > 0 and neighbor != source and neighbor != sink and neighbor not in active:
                    active.append(neighbor)

                pushed = True

                # If no more excess, remove from active list
                if excess[v] == 0:
                    active.pop(0)
                    break

        # If couldn't push, relabel v
        if not pushed and excess[v] > 0:
            min_height = float('inf')
            for u in graph.get_neighbors(v):
                if residual_graph[(v, u)] > 0:
                    min_height = min(min_height, height[u])

            height[v] = min_height + 1

    # Calculate max flow by summing flow out of source
    max_flow = sum(flow.get((source, v), 0) for v in graph.get_neighbors(source))

    # Filter out zero flows and backward edges
    flow_dict = {(u, v): f for (u, v), f in flow.items() if f > 0 and (u, v) in graph.edges}

    return max_flow, flow_dict


def min_cut(graph, source: Any, sink: Any) -> Tuple[Set[Any], Set[Any], List[Tuple[Any, Any]]]:
    """
    Find the minimum s-t cut in a flow network.

    Args:
        graph: The flow network graph
        source: The source vertex
        sink: The sink vertex

    Returns:
        Tuple of (S, T, cut_edges) where:
        - S: Set of vertices on the source side of the cut
        - T: Set of vertices on the sink side of the cut
        - cut_edges: List of edges crossing the cut
    """
    # First, find maximum flow
    max_flow, flow_dict = ford_fulkerson(graph, source, sink)

    # Create residual graph with the final flow
    residual_graph = {}
    for u, v in graph.edges:
        capacity = graph.get_edge_weight(u, v)
        flow = flow_dict.get((u, v), 0)
        residual_graph[(u, v)] = capacity - flow
        residual_graph[(v, u)] = flow

    # Find vertices reachable from source in residual graph
    S = set()
    visited = set()
    queue = collections.deque([source])
    visited.add(source)

    while queue:
        u = queue.popleft()
        S.add(u)

        for v in graph.get_neighbors(u):
            if v not in visited and residual_graph.get((u, v), 0) > 0:
                visited.add(v)
                queue.append(v)

    # Vertices not in S are in T
    T = set(graph.vertices) - S

    # Find edges crossing the cut
    cut_edges = [(u, v) for u in S for v in T if (u, v) in graph.edges]

    return S, T, cut_edges


def bipartite_matching(graph) -> Tuple[Dict[Any, Any], int]:
    """
    Find a maximum bipartite matching using Ford-Fulkerson algorithm.

    Args:
        graph: A bipartite graph with two disjoint vertex sets

    Returns:
        Tuple of (matching, size) where:
        - matching: Dictionary mapping left vertices to their matched right vertices
        - size: Size of the maximum matching
    """
    # Check if graph is bipartite
    is_bipartite, left_set, right_set = _is_bipartite(graph)

    if not is_bipartite:
        raise ValueError("Input graph is not bipartite")

    # Create a flow network
    flow_graph = _create_flow_network_for_matching(graph, left_set, right_set)

    # Add source and sink vertices
    source = "source"
    sink = "sink"

    # Connect source to all vertices in left set
    for u in left_set:
        flow_graph.add_edge(source, u, 1)

    # Connect all vertices in right set to sink
    for v in right_set:
        flow_graph.add_edge(v, sink, 1)

    # Find maximum flow
    max_flow, flow_dict = ford_fulkerson(flow_graph, source, sink)

    # Extract the matching
    matching = {}
    for (u, v), flow in flow_dict.items():
        if u in left_set and v in right_set and flow > 0:
            matching[u] = v

    return matching, len(matching)


# Helper functions

def _create_residual_graph(graph):
    """Create a residual graph from a capacity graph."""
    residual_graph = {}

    # Add forward edges with capacity
    for u, v in graph.edges:
        capacity = graph.get_edge_weight(u, v)
        residual_graph[(u, v)] = capacity

        # Add backward edge with zero capacity if it doesn't exist
        if (v, u) not in graph.edges:
            residual_graph[(v, u)] = 0

    return residual_graph


def _find_augmenting_path_bfs(residual_graph, source, sink):
    """Find an augmenting path using BFS in the residual graph."""
    # Keep track of explored paths
    paths = {source: []}
    queue = collections.deque([source])

    while queue:
        u = queue.popleft()

        # For each neighbor with available capacity
        for v, capacity in residual_graph.items():
            if v[0] == u and capacity > 0 and v[1] not in paths:
                # Extend the path
                paths[v[1]] = paths[u] + [u]

                # If we've reached the sink, construct the full path
                if v[1] == sink:
                    path = paths[v[1]] + [sink]

                    # Calculate the bottleneck capacity
                    bottleneck = float('inf')
                    for i in range(len(path) - 1):
                        bottleneck = min(bottleneck, residual_graph[(path[i], path[i + 1])])

                    return path, bottleneck

                queue.append(v[1])

    return None, 0  # No augmenting path found


def _create_level_graph(residual_graph, source, sink):
    """Create a level graph for Dinic's algorithm using BFS."""
    level = {v: -1 for v in set([u for (u, _) in residual_graph])}
    level[source] = 0

    queue = collections.deque([source])
    while queue:
        u = queue.popleft()

        for (v1, v2), capacity in residual_graph.items():
            if v1 == u and capacity > 0 and level[v2] == -1:
                level[v2] = level[u] + 1
                queue.append(v2)

    return level


def _find_augmenting_path_dfs(residual_graph, u, sink, level, path=None, visited=None):
    """Find an augmenting path using DFS in the level graph."""
    if path is None:
        path = [u]
    if visited is None:
        visited = set([u])

    if u == sink:
        # Calculate the bottleneck capacity
        bottleneck = float('inf')
        for i in range(len(path) - 1):
            bottleneck = min(bottleneck, residual_graph[(path[i], path[i + 1])])

        return path, bottleneck

    for (v1, v2), capacity in residual_graph.items():
        if v1 == u and capacity > 0 and level[v2] == level[u] + 1 and v2 not in visited:
            visited.add(v2)
            path.append(v2)

            result, bottleneck = _find_augmenting_path_dfs(residual_graph, v2, sink, level, path, visited)

            if result is not None:
                return result, bottleneck

            path.pop()
            visited.remove(v2)

    return None, 0


def _is_bipartite(graph):
    """Check if a graph is bipartite and return the two vertex sets."""
    if not graph.vertices:
        return True, set(), set()

    color = {}  # 0 for left set, 1 for right set

    for vertex in graph.vertices:
        if vertex not in color:
            queue = collections.deque([vertex])
            color[vertex] = 0

            while queue:
                u = queue.popleft()

                for v in graph.get_neighbors(u):
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False, set(), set()

    left_set = {v for v, c in color.items() if c == 0}
    right_set = {v for v, c in color.items() if c == 1}

    return True, left_set, right_set


def _create_flow_network_for_matching(graph, left_set, right_set):
    """Create a flow network for bipartite matching."""
    from kosmos.graph.base import Graph

    flow_graph = Graph(directed=True, weighted=True)

    # Add vertices
    for vertex in graph.vertices:
        flow_graph.add_vertex(vertex)

    # Add edges with capacity 1 from left set to right set
    for u in left_set:
        for v in graph.get_neighbors(u):
            if v in right_set:
                flow_graph.add_edge(u, v, 1)

    return flow_graph