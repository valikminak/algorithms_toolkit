from typing import List, Tuple, Set, Any
import collections

def tarjan_scc(graph) -> List[List[Any]]:
    """
    Tarjan's algorithm for finding Strongly Connected Components (SCCs) in a directed graph.

    Args:
        graph: The directed graph

    Returns:
        List of SCCs, where each SCC is a list of vertices
    """
    if not graph.directed:
        raise ValueError("Strongly connected components only apply to directed graphs")

    index_counter = 0
    index = {}  # node -> index
    lowlink = {}  # node -> lowlink
    onstack = set()
    stack = []

    sccs = []

    def strongconnect(node):
        nonlocal index_counter

        # Set the depth index for this node
        index[node] = index_counter
        lowlink[node] = index_counter
        index_counter += 1
        stack.append(node)
        onstack.add(node)

        # Consider successors
        for successor in graph.get_neighbors(node):
            if successor not in index:
                # Successor has not yet been visited; recurse on it
                strongconnect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in onstack:
                # Successor is in stack and hence in the current SCC
                lowlink[node] = min(lowlink[node], index[successor])

        # If node is a root node, pop the stack and generate an SCC
        if lowlink[node] == index[node]:
            scc = []

            while True:
                successor = stack.pop()
                onstack.remove(successor)
                scc.append(successor)
                if successor == node:
                    break

            sccs.append(scc)

    # Start DFS from each unvisited node
    for node in graph.vertices:
        if node not in index:
            strongconnect(node)

    return sccs


def articulation_points(graph) -> Set[Any]:
    """
    Find articulation points (cut vertices) in an undirected graph.

    Args:
        graph: The undirected graph

    Returns:
        Set of articulation points
    """
    if graph.directed:
        raise ValueError("Articulation points algorithm requires an undirected graph")

    if not graph.vertices:
        return set()

    disc = {}  # Discovery time
    low = {}  # Earliest visited vertex reachable from subtree
    visited = set()
    ap = set()  # Articulation points
    parent = {}  # Parent in DFS tree
    time = 0

    def dfs(u):
        nonlocal time

        # Count of children in DFS tree
        children = 0

        # Mark the current node as visited
        visited.add(u)

        # Initialize discovery time and low value
        time += 1
        disc[u] = low[u] = time

        # Go through all neighbors
        for v in graph.get_neighbors(u):
            # If v is not visited yet, then make it a child of u
            if v not in visited:
                parent[v] = u
                children += 1
                dfs(v)

                # Check if subtree rooted with v has a connection to ancestor of u
                low[u] = min(low[u], low[v])

                # u is an articulation point in the following cases
                # Case 1: u is root of DFS tree and has two or more children
                if parent.get(u) is None and children > 1:
                    ap.add(u)

                # Case 2: u is not root and low value of one of its children is >= discovery value of u
                if parent.get(u) is not None and low[v] >= disc[u]:
                    ap.add(u)

            # Update low value if v is not parent
            elif v != parent.get(u, None):
                low[u] = min(low[u], disc[v])

    # Call the recursive helper function for all vertices
    for vertex in graph.vertices:
        if vertex not in visited:
            dfs(vertex)

    return ap


def bridges(graph) -> List[Tuple[Any, Any]]:
    """
    Find bridges in an undirected graph.

    Args:
        graph: The undirected graph

    Returns:
        List of bridges as (u, v) tuples
    """
    if graph.directed:
        raise ValueError("Bridges algorithm requires an undirected graph")

    if not graph.vertices:
        return []

    disc = {}  # Discovery time
    low = {}  # Earliest visited vertex reachable from subtree
    visited = set()
    bridges_list = []
    parent = {}  # Parent in DFS tree
    time = 0

    def dfs(u):
        nonlocal time

        # Mark the current node as visited
        visited.add(u)

        # Initialize discovery time and low value
        time += 1
        disc[u] = low[u] = time

        # Go through all neighbors
        for v in graph.get_neighbors(u):
            # If v is not visited yet, then make it a child of u
            if v not in visited:
                parent[v] = u
                dfs(v)

                # Check if subtree rooted with v has a connection to ancestor of u
                low[u] = min(low[u], low[v])

                # If the lowest vertex reachable from subtree under v is below u,
                # then u-v is a bridge
                if low[v] > disc[u]:
                    bridges_list.append((u, v))

            # Update low value if v is not parent
            elif v != parent.get(u, None):
                low[u] = min(low[u], disc[v])

    # Call the recursive helper function for all vertices
    for vertex in graph.vertices:
        if vertex not in visited:
            dfs(vertex)

    return bridges_list


def has_eulerian_path(graph) -> bool:
    """
    Check if a graph has an Eulerian path.

    Args:
        graph: The graph

    Returns:
        True if the graph has an Eulerian path, False otherwise
    """
    if not graph.vertices:
        return True

    # Count degrees
    degree = {v: 0 for v in graph.vertices}
    for u, v in graph.edges:
        degree[u] += 1
        if not graph.directed:
            degree[v] += 1
        else:
            # For directed graphs, we need to separately count in-degree and out-degree
            if v not in degree:
                degree[v] = 0

    if graph.directed:
        # For directed graphs, check in-degree and out-degree
        in_degree = {v: 0 for v in graph.vertices}
        out_degree = {v: 0 for v in graph.vertices}

        for u, v in graph.edges:
            out_degree[u] += 1
            in_degree[v] += 1

        # Count vertices with in_degree != out_degree
        odd_vertices = sum(1 for v in graph.vertices if in_degree[v] != out_degree[v])

        # Either all vertices have equal in-degree and out-degree, or
        # exactly one vertex has out_degree - in_degree = 1 and
        # exactly one vertex has in_degree - out_degree = 1
        return odd_vertices == 0 or (
                odd_vertices == 2 and
                sum(1 for v in graph.vertices if out_degree[v] - in_degree[v] == 1) == 1 and
                sum(1 for v in graph.vertices if in_degree[v] - out_degree[v] == 1) == 1
        )
    else:
        # For undirected graphs, either all vertices have even degree, or
        # exactly two vertices have odd degree
        odd_vertices = sum(1 for v in graph.vertices if degree[v] % 2 == 1)
        return odd_vertices == 0 or odd_vertices == 2


def has_eulerian_circuit(graph) -> bool:
    """
    Check if a graph has an Eulerian circuit.

    Args:
        graph: The graph

    Returns:
        True if the graph has an Eulerian circuit, False otherwise
    """
    if not graph.vertices:
        return True

    if graph.directed:
        # For directed graphs, check if every vertex has equal in-degree and out-degree
        in_degree = {v: 0 for v in graph.vertices}
        out_degree = {v: 0 for v in graph.vertices}

        for u, v in graph.edges:
            out_degree[u] += 1
            in_degree[v] += 1

        # Every vertex must have equal in-degree and out-degree
        return all(in_degree[v] == out_degree[v] for v in graph.vertices)
    else:
        # For undirected graphs, check if every vertex has even degree
        degree = {v: 0 for v in graph.vertices}
        for u, v in graph.edges:
            degree[u] += 1
            degree[v] += 1

        return all(degree[v] % 2 == 0 for v in graph.vertices)


def find_eulerian_path(graph) -> List[Any]:
    """
    Find an Eulerian path in a graph.

    Args:
        graph: The graph

    Returns:
        List of vertices forming an Eulerian path, or empty list if none exists
    """
    if not has_eulerian_path(graph):
        return []

    if not graph.vertices:
        return []

    # For directed graphs, find a vertex with out_degree > in_degree if it exists
    # For undirected graphs, find a vertex with odd degree if it exists
    start_vertex = None

    if graph.directed:
        in_degree = {v: 0 for v in graph.vertices}
        out_degree = {v: 0 for v in graph.vertices}

        for u, v in graph.edges:
            out_degree[u] += 1
            in_degree[v] += 1

        for v in graph.vertices:
            if out_degree[v] - in_degree[v] == 1:
                start_vertex = v
                break
    else:
        degree = {v: 0 for v in graph.vertices}
        for u, v in graph.edges:
            degree[u] += 1
            degree[v] += 1

        for v in graph.vertices:
            if degree[v] % 2 == 1:
                start_vertex = v
                break

    # If no suitable starting vertex was found, use any vertex
    if start_vertex is None:
        start_vertex = next(iter(graph.vertices))

    # Make a copy of the graph and find the Eulerian path
    temp_graph = graph.__class__(directed=graph.directed, weighted=graph.weighted)
    for vertex in graph.vertices:
        temp_graph.add_vertex(vertex)
    for edge in graph.edges:
        temp_graph.add_edge(edge[0], edge[1], graph.edges[edge])

    path = []

    def dfs(v):
        while temp_graph.get_neighbors(v):
            u = temp_graph.get_neighbors(v)[0]
            temp_graph.remove_edge(v, u)
            dfs(u)
        path.append(v)

    dfs(start_vertex)
    path.reverse()

    # Verify the path is valid
    if len(path) != len(graph.edges) + 1:
        return []  # Not a valid Eulerian path

    return path


def hierholzer_eulerian_circuit(graph) -> List[Any]:
    """
    Find an Eulerian circuit in a graph using Hierholzer's algorithm.

    Args:
        graph: The graph

    Returns:
        List of vertices forming an Eulerian circuit, or empty list if none exists
    """
    if not has_eulerian_circuit(graph):
        return []

    if not graph.vertices:
        return []

    # Make a copy of the graph
    temp_graph = graph.__class__(directed=graph.directed, weighted=graph.weighted)
    for vertex in graph.vertices:
        temp_graph.add_vertex(vertex)
    for edge in graph.edges:
        temp_graph.add_edge(edge[0], edge[1], graph.edges[edge])

    # Start from any vertex
    start_vertex = next(iter(graph.vertices))

    # Find a circuit
    circuit = []

    def find_circuit(start):
        current = start
        path = [current]

        while temp_graph.get_neighbors(current):
            neighbor = temp_graph.get_neighbors(current)[0]
            temp_graph.remove_edge(current, neighbor)
            current = neighbor
            path.append(current)

        return path

    # Find initial circuit
    current_circuit = find_circuit(start_vertex)
    circuit = current_circuit

    # Check for remaining edges
    i = 0
    while i < len(circuit):
        vertex = circuit[i]
        if temp_graph.get_neighbors(vertex):
            # Find a new circuit starting from vertex
            new_circuit = find_circuit(vertex)

            # Insert the new circuit into the existing one
            circuit = circuit[:i] + new_circuit[:-1] + circuit[i:]
        else:
            i += 1

    return circuit


def is_connected(graph) -> bool:
    """
    Check if an undirected graph is connected.

    Args:
        graph: The undirected graph

    Returns:
        True if the graph is connected, False otherwise
    """
    if not graph.vertices:
        return True

    if graph.directed:
        # For directed graphs, use strong connectivity
        return len(tarjan_scc(graph)) == 1

    # For undirected graphs, use BFS to check connectivity
    start = next(iter(graph.vertices))
    visited = set()
    queue = collections.deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(graph.vertices)


def connected_components(graph) -> List[Set[Any]]:
    """
    Find all connected components in an undirected graph.

    Args:
        graph: The undirected graph

    Returns:
        List of sets, each set containing the vertices of a connected component
    """
    if graph.directed:
        raise ValueError("Connected components algorithm requires an undirected graph")

    if not graph.vertices:
        return []

    components = []
    visited = set()

    for vertex in graph.vertices:
        if vertex in visited:
            continue

        component = set()
        queue = collections.deque([vertex])
        component.add(vertex)
        visited.add(vertex)

        while queue:
            v = queue.popleft()

            for neighbor in graph.get_neighbors(v):
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return components