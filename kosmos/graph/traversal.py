from collections import deque


def breadth_first_search(graph, start_vertex):
    """
    Breadth-first search algorithm.

    Args:
        graph: The graph to search
        start_vertex: Starting vertex

    Returns:
        Dictionary of visited vertices
    """
    visited = {}
    queue = deque([start_vertex])
    visited[start_vertex] = True

    while queue:
        vertex = queue.popleft()

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited[neighbor] = True
                queue.append(neighbor)

    return visited


def depth_first_search(graph, start_vertex):
    """
    Depth-first search algorithm.

    Args:
        graph: The graph to search
        start_vertex: Starting vertex

    Returns:
        Dictionary of visited vertices
    """
    visited = {}

    def dfs_visit(vertex):
        visited[vertex] = True

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs_visit(neighbor)

    dfs_visit(start_vertex)
    return visited


def iterative_dfs(graph, start_vertex):
    """
    Iterative implementation of depth-first search.

    Args:
        graph: The graph to search
        start_vertex: Starting vertex

    Returns:
        Dictionary of visited vertices
    """
    visited = {}
    stack = [start_vertex]

    while stack:
        vertex = stack.pop()

        if vertex not in visited:
            visited[vertex] = True

            # Add neighbors in reverse order to maintain DFS order
            neighbors = list(graph.get_neighbors(vertex))
            neighbors.reverse()

            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)

    return visited


def topological_sort(graph):
    """
    Topological sort of a directed acyclic graph (DAG).

    Args:
        graph: The directed graph

    Returns:
        List of vertices in topological order
    """
    if not graph.directed:
        raise ValueError("Topological sort requires a directed graph")

    visited = set()
    temp_marked = set()  # For cycle detection
    order = []

    def visit(vertex):
        if vertex in temp_marked:
            raise ValueError("Graph contains a cycle")

        if vertex not in visited:
            temp_marked.add(vertex)

            for neighbor in graph.get_neighbors(vertex):
                visit(neighbor)

            temp_marked.remove(vertex)
            visited.add(vertex)
            order.append(vertex)

    for vertex in graph.vertices:
        if vertex not in visited:
            visit(vertex)

    return list(reversed(order))


def is_cyclic(graph):
    """
    Check if a graph contains a cycle.

    Args:
        graph: The graph to check

    Returns:
        True if the graph contains a cycle, False otherwise
    """
    visited = set()
    rec_stack = set()

    def is_cyclic_util(vertex):
        visited.add(vertex)
        rec_stack.add(vertex)

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                if is_cyclic_util(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(vertex)
        return False

    # For directed graph
    if graph.directed:
        for vertex in graph.vertices:
            if vertex not in visited:
                if is_cyclic_util(vertex):
                    return True
        return False
    # For undirected graph
    else:
        parent = {}

        def is_cyclic_undirected(vertex, parent_vertex=None):
            visited.add(vertex)

            for neighbor in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    parent[neighbor] = vertex
                    if is_cyclic_undirected(neighbor, vertex):
                        return True
                elif parent_vertex != neighbor:
                    return True

            return False

        for vertex in graph.vertices:
            if vertex not in visited:
                if is_cyclic_undirected(vertex):
                    return True

        return False