from typing import List, Dict, Tuple, Set, Optional, Any, Callable, Union, Generic, TypeVar
import collections


def breadth_first_search(graph, start: Any) -> Dict[Any, Optional[Any]]:
    """
    Perform Breadth-First Search on a graph.

    Args:
        graph: The graph to search
        start: The starting vertex

    Returns:
        Dictionary mapping each reachable vertex to its parent in the BFS tree.
        Unreachable vertices are not included.
    """
    if start not in graph.vertices:
        return {}

    parent = {start: None}
    queue = collections.deque([start])
    visited = {start}

    while queue:
        vertex = queue.popleft()

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = vertex
                queue.append(neighbor)

    return parent


def depth_first_search(graph, start: Any = None) -> Dict[Any, Tuple[int, int, Optional[Any]]]:
    """
    Perform Depth-First Search on a graph.

    Args:
        graph: The graph to search
        start: Optional starting vertex. If None, DFS is performed on all components.

    Returns:
        Dictionary mapping each vertex to its discovery time, finish time, and parent.
        Format: {vertex: (discovery_time, finish_time, parent)}
    """
    result = {}
    time = 0

    def dfs_visit(vertex, parent=None):
        nonlocal time
        time += 1
        discovery_time = time
        result[vertex] = (discovery_time, None, parent)

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in result:
                dfs_visit(neighbor, vertex)

        time += 1
        finish_time = time
        result[vertex] = (discovery_time, finish_time, parent)

    if start is not None:
        if start in graph.vertices:
            dfs_visit(start)
    else:
        # Perform DFS on all components
        for vertex in graph.vertices:
            if vertex not in result:
                dfs_visit(vertex)

    return result


def topological_sort(graph) -> List[Any]:
    """
    Perform a topological sort on a directed acyclic graph (DAG).

    Args:
        graph: The directed graph to sort

    Returns:
        List of vertices in topological order. If graph has a cycle, returns an empty list.
    """
    if not graph.directed:
        raise ValueError("Topological sort only applies to directed graphs")

    # Check for cycles using DFS
    visited = set()
    temp_visited = set()  # For cycle detection
    order = []

    def visit(vertex):
        if vertex in temp_visited:
            # Cycle detected
            return False

        if vertex in visited:
            return True

        temp_visited.add(vertex)

        for neighbor in graph.get_neighbors(vertex):
            if not visit(neighbor):
                return False

        temp_visited.remove(vertex)
        visited.add(vertex)
        order.append(vertex)
        return True

    for vertex in graph.vertices:
        if vertex not in visited:
            if not visit(vertex):
                return []  # Cycle detected

    return list(reversed(order))


def iterative_dfs(graph, start: Any) -> Dict[Any, Tuple[int, int, Optional[Any]]]:
    """
    Perform Depth-First Search iteratively (non-recursive).

    Args:
        graph: The graph to search
        start: The starting vertex

    Returns:
        Dictionary mapping each vertex to its discovery time, finish time, and parent.
        Format: {vertex: (discovery_time, finish_time, parent)}
    """
    if start not in graph.vertices:
        return {}

    result = {}
    time = 0
    stack = [(start, None, False)]  # (vertex, parent, is_finished)
    visited = set()

    while stack:
        vertex, parent, is_finished = stack.pop()

        if is_finished:
            # Finishing visit
            time += 1
            finish_time = time
            discovery_time, _, _ = result[vertex]
            result[vertex] = (discovery_time, finish_time, parent)
        elif vertex not in visited:
            # Starting visit
            visited.add(vertex)
            time += 1
            discovery_time = time
            result[vertex] = (discovery_time, None, parent)

            # Add finishing visit to stack
            stack.append((vertex, parent, True))

            # Add neighbors to stack (in reverse order to match recursive DFS)
            neighbors = list(graph.get_neighbors(vertex))
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append((neighbor, vertex, False))

    return result


def dfs_with_callback(graph, start: Any, pre_visit: Optional[Callable] = None,
                      post_visit: Optional[Callable] = None) -> Set[Any]:
    """
    Perform DFS with customizable callbacks for pre-visit and post-visit actions.

    Args:
        graph: The graph to search
        start: The starting vertex
        pre_visit: Function to call before exploring a vertex's neighbors
        post_visit: Function to call after exploring a vertex's neighbors

    Returns:
        Set of visited vertices
    """
    visited = set()

    def dfs_visit(vertex):
        visited.add(vertex)

        if pre_visit:
            pre_visit(vertex)

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs_visit(neighbor)

        if post_visit:
            post_visit(vertex)

    if start in graph.vertices:
        dfs_visit(start)

    return visited


def bfs_with_distance(graph, start: Any) -> Tuple[Dict[Any, int], Dict[Any, Optional[Any]]]:
    """
    Perform BFS and calculate distances from the start vertex.

    Args:
        graph: The graph to search
        start: The starting vertex

    Returns:
        Tuple of (distances, parents) where:
        - distances: Dictionary mapping each vertex to its distance from start
        - parents: Dictionary mapping each vertex to its parent in the BFS tree
    """
    if start not in graph.vertices:
        return {}, {}

    distances = {start: 0}
    parents = {start: None}
    queue = collections.deque([start])
    visited = {start}

    while queue:
        vertex = queue.popleft()
        dist = distances[vertex]

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = dist + 1
                parents[neighbor] = vertex
                queue.append(neighbor)

    return distances, parents


def find_all_paths(graph, start: Any, end: Any, path: Optional[List[Any]] = None) -> List[List[Any]]:
    """
    Find all paths between two vertices in a graph using DFS.

    Args:
        graph: The graph to search
        start: Starting vertex
        end: Ending vertex
        path: Current path (used in recursion)

    Returns:
        List of all paths from start to end
    """
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return [path]

    if start not in graph.vertices:
        return []

    paths = []
    for neighbor in graph.get_neighbors(start):
        if neighbor not in path:  # Avoid cycles
            new_paths = find_all_paths(graph, neighbor, end, path)
            paths.extend(new_paths)

    return paths


def find_cycle(graph) -> Optional[List[Any]]:
    """
    Find a cycle in the graph if one exists.

    Args:
        graph: The graph to search

    Returns:
        A list of vertices forming a cycle, or None if no cycle exists
    """
    visited = set()
    rec_stack = set()

    def dfs_cycle(vertex, parent):
        visited.add(vertex)
        rec_stack.add(vertex)

        for neighbor in graph.get_neighbors(vertex):
            # Skip the edge to parent in undirected graphs
            if graph.directed == False and neighbor == parent:
                continue

            if neighbor not in visited:
                cycle = dfs_cycle(neighbor, vertex)
                if cycle:
                    return cycle
            elif neighbor in rec_stack:
                # Found a cycle
                cycle = []
                curr = vertex
                while curr != neighbor:
                    cycle.append(curr)
                    # Need to find the parent - in real implementation,
                    # we would track the path to each vertex
                    for v in graph.vertices:
                        if v in visited and curr in graph.get_neighbors(v):
                            curr = v
                            break
                cycle.append(neighbor)
                cycle.append(vertex)  # Complete the cycle
                return cycle

        rec_stack.remove(vertex)
        return None

    for vertex in graph.vertices:
        if vertex not in visited:
            cycle = dfs_cycle(vertex, None)
            if cycle:
                return cycle

    return None