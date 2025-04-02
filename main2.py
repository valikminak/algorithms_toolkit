import os
import time
import math
import random
import heapq
import bisect
import functools
import itertools
import collections
from typing import List, Dict, Tuple, Set, Optional, Any, Callable, Union, Generic, TypeVar
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# Type variables for generic implementations
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


###########################################
# Graph Data Structures and Algorithms
###########################################

class Graph:
    """Flexible graph implementation supporting directed/undirected and weighted/unweighted graphs."""

    def __init__(self, directed: bool = False, weighted: bool = False):
        """
        Initialize a graph.

        Args:
            directed: If True, the graph is directed, otherwise undirected.
            weighted: If True, the graph supports edge weights.
        """
        self.directed = directed
        self.weighted = weighted
        self.vertices = set()
        self.edges = {}  # {(u, v): weight} for weighted or {(u, v): None} for unweighted
        self.adj_list = collections.defaultdict(list)

    def add_vertex(self, vertex: Any) -> None:
        """Add a vertex to the graph."""
        self.vertices.add(vertex)
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, u: Any, v: Any, weight: Optional[float] = None) -> None:
        """
        Add an edge to the graph.

        Args:
            u: Source vertex
            v: Target vertex
            weight: Edge weight (only used if graph is weighted)
        """
        # Add vertices if they don't exist
        self.add_vertex(u)
        self.add_vertex(v)

        # Add edge
        if self.weighted and weight is None:
            weight = 1.0  # Default weight

        if self.weighted:
            self.edges[(u, v)] = weight
        else:
            self.edges[(u, v)] = None

        self.adj_list[u].append(v)

        # If undirected, add the reverse edge
        if not self.directed:
            if self.weighted:
                self.edges[(v, u)] = weight
            else:
                self.edges[(v, u)] = None

            self.adj_list[v].append(u)

    def remove_edge(self, u: Any, v: Any) -> None:
        """Remove an edge from the graph."""
        if (u, v) in self.edges:
            del self.edges[(u, v)]
            self.adj_list[u].remove(v)

            if not self.directed and (v, u) in self.edges:
                del self.edges[(v, u)]
                self.adj_list[v].remove(u)

    def remove_vertex(self, vertex: Any) -> None:
        """Remove a vertex and all its connected edges from the graph."""
        if vertex in self.vertices:
            self.vertices.remove(vertex)

            # Remove all edges connected to this vertex
            edges_to_remove = [(u, v) for (u, v) in self.edges if u == vertex or v == vertex]
            for u, v in edges_to_remove:
                self.remove_edge(u, v)

            # Remove from adjacency list
            del self.adj_list[vertex]
            for v in self.adj_list:
                self.adj_list[v] = [u for u in self.adj_list[v] if u != vertex]

    def get_neighbors(self, vertex: Any) -> List[Any]:
        """Get all neighbors of a vertex."""
        return self.adj_list[vertex]

    def get_edge_weight(self, u: Any, v: Any) -> Optional[float]:
        """Get the weight of an edge."""
        if (u, v) in self.edges:
            return self.edges[(u, v)]
        return None

    def get_edges(self) -> List[Tuple[Any, Any, Optional[float]]]:
        """Get all edges in the graph as (u, v, weight) tuples."""
        return [(u, v, w) for (u, v), w in self.edges.items()]

    def get_vertices(self) -> Set[Any]:
        """Get all vertices in the graph."""
        return self.vertices

    def __str__(self) -> str:
        """String representation of the graph."""
        result = f"{'Directed' if self.directed else 'Undirected'} "
        result += f"{'weighted' if self.weighted else 'unweighted'} graph with "
        result += f"{len(self.vertices)} vertices and {len(self.edges)} edges.\n"

        for vertex in sorted(self.vertices):
            result += f"{vertex}: "
            neighbors = []
            for neighbor in self.adj_list[vertex]:
                if self.weighted:
                    weight = self.edges.get((vertex, neighbor))
                    neighbors.append(f"{neighbor}({weight})")
                else:
                    neighbors.append(f"{neighbor}")
            result += ", ".join(neighbors) + "\n"

        return result


class DisjointSet:
    """
    Disjoint Set (Union-Find) data structure with path compression and union by rank.
    Used for efficient checking if two elements are in the same set.
    """

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x: Any) -> None:
        """Create a new set with a single element x."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: Any) -> Any:
        """Find the representative (root) of the set containing element x."""
        # Create set if needed
        if x not in self.parent:
            self.make_set(x)

        # Path compression: make the found root the direct parent
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: Any, y: Any) -> None:
        """Merge the sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return  # Already in the same set

        # Union by rank: attach the smaller tree to the root of the larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            # Same rank, arbitrarily choose root_x as the new root
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def connected(self, x: Any, y: Any) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def get_sets(self) -> Dict[Any, List[Any]]:
        """Return all disjoint sets as a dictionary of representatives to element lists."""
        result = collections.defaultdict(list)
        for x in self.parent:
            result[self.find(x)].append(x)
        return dict(result)


def breadth_first_search(graph: Graph, start: Any) -> Dict[Any, Optional[Any]]:
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


def depth_first_search(graph: Graph, start: Any = None) -> Dict[Any, Tuple[int, int, Optional[Any]]]:
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


def topological_sort(graph: Graph) -> List[Any]:
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


def dijkstra(graph: Graph, start: Any) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]]]:
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


def bellman_ford(graph: Graph, start: Any) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]], bool]:
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


def floyd_warshall(graph: Graph) -> Tuple[Dict[Tuple[Any, Any], float], Dict[Tuple[Any, Any], Optional[Any]]]:
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


def kruskal_mst(graph: Graph) -> List[Tuple[Any, Any, float]]:
    """
    Kruskal's algorithm for finding a Minimum Spanning Tree (MST).

    Args:
        graph: The weighted graph

    Returns:
        List of edges in the MST as (u, v, weight) tuples
    """
    if not graph.weighted:
        raise ValueError("Kruskal's algorithm requires a weighted graph")

    if graph.directed:
        raise ValueError("Kruskal's algorithm requires an undirected graph")

    # Sort edges by weight
    edges = graph.get_edges()
    edges.sort(key=lambda x: x[2])

    # Initialize disjoint set
    ds = DisjointSet()
    for vertex in graph.vertices:
        ds.make_set(vertex)

    # Initialize MST
    mst = []

    # Process edges in order of increasing weight
    for u, v, weight in edges:
        # If including this edge doesn't create a cycle, add it to the MST
        if not ds.connected(u, v):
            ds.union(u, v)
            mst.append((u, v, weight))

            # Early stopping when MST is complete
            if len(mst) == len(graph.vertices) - 1:
                break

    return mst


def prim_mst(graph: Graph, start: Any = None) -> List[Tuple[Any, Any, float]]:
    """
    Prim's algorithm for finding a Minimum Spanning Tree (MST).

    Args:
        graph: The weighted graph
        start: Optional starting vertex. If None, a random vertex is chosen.

    Returns:
        List of edges in the MST as (u, v, weight) tuples
    """
    if not graph.weighted:
        raise ValueError("Prim's algorithm requires a weighted graph")

    if graph.directed:
        raise ValueError("Prim's algorithm requires an undirected graph")

    if len(graph.vertices) == 0:
        return []

    # Choose a starting vertex if not provided
    if start is None or start not in graph.vertices:
        start = next(iter(graph.vertices))

    # Initialize
    mst = []
    visited = {start}

    # Priority queue of (weight, u, v) where u is in the MST and v is not
    edges = []
    for neighbor in graph.get_neighbors(start):
        weight = graph.get_edge_weight(start, neighbor)
        heapq.heappush(edges, (weight, start, neighbor))

    # Grow the MST
    while edges and len(visited) < len(graph.vertices):
        weight, u, v = heapq.heappop(edges)

        if v in visited:
            continue

        # Add vertex v to the MST
        visited.add(v)
        mst.append((u, v, weight))

        # Add edges from v to unvisited vertices
        for neighbor in graph.get_neighbors(v):
            if neighbor not in visited:
                weight = graph.get_edge_weight(v, neighbor)
                heapq.heappush(edges, (weight, v, neighbor))

    return mst


def tarjan_scc(graph: Graph) -> List[List[Any]]:
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


def articulation_points(graph: Graph) -> Set[Any]:
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


def bridges(graph: Graph) -> List[Tuple[Any, Any]]:
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


def has_eulerian_path(graph: Graph) -> bool:
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


def has_eulerian_circuit(graph: Graph) -> bool:
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


def find_eulerian_path(graph: Graph) -> List[Any]:
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
    temp_graph = Graph(directed=graph.directed, weighted=graph.weighted)
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
        path.appenpath.append(v)

    dfs(start_vertex)
    path.reverse()

    # Verify the path is valid
    if len(path) != len(graph.edges) + 1:
        return []  # Not a valid Eulerian path

    return path


def hierholzer_eulerian_circuit(graph: Graph) -> List[Any]:
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
    temp_graph = Graph(directed=graph.directed, weighted=graph.weighted)
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


###########################################
# Tree Data Structures and Algorithms
###########################################

class TreeNode:
    """Basic tree node implementation."""

    def __init__(self, value: Any):
        self.value = value
        self.children = []

    def add_child(self, child: 'TreeNode') -> None:
        """Add a child node."""
        self.children.append(child)

    def __str__(self) -> str:
        return str(self.value)


class BinaryTreeNode:
    """Binary tree node implementation."""

    def __init__(self, value: Any):
        self.value = value
        self.left = None
        self.right = None

    def __str__(self) -> str:
        return str(self.value)


def tree_height(root: TreeNode) -> int:
    """
    Calculate the height of a tree.

    Args:
        root: The root node of the tree

    Returns:
        The height of the tree (0 for empty tree)
    """
    if root is None:
        return 0

    if not root.children:
        return 1

    return 1 + max(tree_height(child) for child in root.children)


def binary_tree_height(root: BinaryTreeNode) -> int:
    """
    Calculate the height of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        The height of the binary tree (0 for empty tree)
    """
    if root is None:
        return 0

    return 1 + max(binary_tree_height(root.left), binary_tree_height(root.right))


def binary_tree_size(root: BinaryTreeNode) -> int:
    """
    Calculate the size (number of nodes) of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        The number of nodes in the binary tree
    """
    if root is None:
        return 0

    return 1 + binary_tree_size(root.left) + binary_tree_size(root.right)


def binary_tree_is_balanced(root: BinaryTreeNode) -> bool:
    """
    Check if a binary tree is balanced (the height difference between left and right subtrees is at most 1).

    Args:
        root: The root node of the binary tree

    Returns:
        True if the binary tree is balanced, False otherwise
    """
    def check_height(node):
        if node is None:
            return 0

        left_height = check_height(node.left)
        if left_height == -1:
            return -1  # Left subtree is unbalanced

        right_height = check_height(node.right)
        if right_height == -1:
            return -1  # Right subtree is unbalanced

        if abs(left_height - right_height) > 1:
            return -1  # Current node is unbalanced

        return 1 + max(left_height, right_height)

    return check_height(root) != -1


def binary_tree_inorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform inorder traversal of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in inorder traversal order
    """
    result = []

    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.value)
            inorder(node.right)

    inorder(root)
    return result


def binary_tree_preorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform preorder traversal of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in preorder traversal order
    """
    result = []

    def preorder(node):
        if node:
            result.append(node.value)
            preorder(node.left)
            preorder(node.right)

    preorder(root)
    return result


def binary_tree_postorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform postorder traversal of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in postorder traversal order
    """
    result = []

    def postorder(node):
        if node:
            postorder(node.left)
            postorder(node.right)
            result.append(node.value)

    postorder(root)
    return result


def binary_tree_levelorder_traversal(root: BinaryTreeNode) -> List[List[Any]]:
    """
    Perform level-order traversal of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        List of lists, where each inner list contains node values at the same level
    """
    if not root:
        return []

    result = []
    queue = collections.deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.value)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result


def binary_tree_is_bst(root: BinaryTreeNode) -> bool:
    """
    Check if a binary tree is a valid binary search tree (BST).

    Args:
        root: The root node of the binary tree

    Returns:
        True if the binary tree is a valid BST, False otherwise
    """
    def is_valid_bst(node, min_val=float('-inf'), max_val=float('inf')):
        if node is None:
            return True

        if node.value <= min_val or node.value >= max_val:
            return False

        return (is_valid_bst(node.left, min_val, node.value) and
                is_valid_bst(node.right, node.value, max_val))

    return is_valid_bst(root)


def binary_tree_lowest_common_ancestor(root: BinaryTreeNode, p: BinaryTreeNode, q: BinaryTreeNode) -> Optional[BinaryTreeNode]:
    """
    Find the lowest common ancestor of two nodes in a binary tree.

    Args:
        root: The root node of the binary tree
        p, q: The two nodes to find the lowest common ancestor for

    Returns:
        The lowest common ancestor node, or None if not found
    """
    if not root or root == p or root == q:
        return root

    left = binary_tree_lowest_common_ancestor(root.left, p, q)
    right = binary_tree_lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root
    return left if left else right


def binary_tree_serialize(root: BinaryTreeNode) -> str:
    """
    Serialize a binary tree to a string.

    Args:
        root: The root node of the binary tree

    Returns:
        String representation of the binary tree
    """
    if not root:
        return "null"

    return (str(root.value) + "," +
            binary_tree_serialize(root.left) + "," +
            binary_tree_serialize(root.right))


def binary_tree_deserialize(data: str) -> Optional[BinaryTreeNode]:
    """
    Deserialize a string to a binary tree.

    Args:
        data: String representation of the binary tree

    Returns:
        The root node of the deserialized binary tree
    """
    def deserialize_helper(nodes):
        if not nodes:
            return None

        value = nodes.popleft()
        if value == "null":
            return None

        node = BinaryTreeNode(value)
        node.left = deserialize_helper(nodes)
        node.right = deserialize_helper(nodes)

        return node

    nodes = collections.deque(data.split(","))
    return deserialize_helper(nodes)


###########################################
# String Algorithms
###########################################

def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome (reads the same forward and backward).

    Args:
        s: The string to check

    Returns:
        True if the string is a palindrome, False otherwise
    """
    # Convert to lowercase and remove non-alphanumeric characters
    s = ''.join(c.lower() for c in s if c.isalnum())

    # Check if the string is equal to its reverse
    return s == s[::-1]


def longest_palindromic_substring(s: str) -> str:
    """
    Find the longest palindromic substring in a string.

    Args:
        s: The input string

    Returns:
        The longest palindromic substring
    """
    if not s:
        return ""

    start = 0
    max_length = 1

    # Helper function to expand around center
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    for i in range(len(s)):
        # Expand around center (odd length)
        length1 = expand_around_center(i, i)

        # Expand around center (even length)
        length2 = expand_around_center(i, i + 1)

        # Get maximum length
        length = max(length1, length2)

        # Update result if needed
        if length > max_length:
            max_length = length
            start = i - (length - 1) // 2

    return s[start:start + max_length]


def kmp_search(text: str, pattern: str) -> List[int]:
    """
    Knuth-Morris-Pratt (KMP) algorithm for pattern matching in strings.

    Args:
        text: The text to search in
        pattern: The pattern to search for

    Returns:
        List of starting indices where the pattern is found in the text
    """
    if not pattern:
        return []

    # Compute the LPS (Longest Proper Prefix which is also Suffix) array
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1

        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        return lps

    lps = compute_lps(pattern)

    # Search for the pattern
    results = []
    i = 0  # Index for text
    j = 0  # Index for pattern

    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == len(pattern):
            # Pattern found at index i - j
            results.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results


def rabin_karp_search(text: str, pattern: str, q: int = 101) -> List[int]:
    """
    Rabin-Karp algorithm for pattern matching in strings.

    Args:
        text: The text to search in
        pattern: The pattern to search for
        q: A prime number used for hashing

    Returns:
        List of starting indices where the pattern is found in the text
    """
    if not pattern:
        return []

    n = len(text)
    m = len(pattern)
    results = []

    if m > n:
        return results

    # Hash function: h(s) = (s[0] * d^(m-1) + s[1] * d^(m-2) + ... + s[m-1]) % q
    # where d is the number of characters in the alphabet
    d = 256  # Assuming ASCII

    # Calculate (d^(m-1)) % q
    h = 1
    for _ in range(m - 1):
        h = (h * d) % q

    # Calculate hash values for pattern and first window of text
    p = 0  # Hash value for pattern
    t = 0  # Hash value for current window of text

    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    # Slide the pattern over text one by one
    for i in range(n - m + 1):
        # Check if hash values match
        if p == t:
            # Check if the actual pattern matches
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break

            if match:
                results.append(i)

        # Calculate hash value for next window
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q

            # We might get negative value, convert it to positive
            if t < 0:
                t += q

    return results


def z_algorithm(s: str) -> List[int]:
    """
    Z Algorithm for pattern matching.

    Args:
        s: The input string

    Returns:
        Z array where Z[i] is the length of the longest substring starting from s[i]
        which is also a prefix of s
    """
    n = len(s)
    z = [0] * n

    # Initial window
    left, right = 0, 0

    for i in range(1, n):
        # If i is outside the current window, compute Z[i] naively
        if i > right:
            left = right = i

            # Check if s[left...] matches with s[0...]
            while right < n and s[right] == s[right - left]:
                right += 1

            z[i] = right - left
            right -= 1
        else:
            # We are within the window, copy values
            k = i - left

            # If the value we're copying doesn't hit the window boundary, just copy
            if z[k] < right - i + 1:
                z[i] = z[k]
            else:
                # Otherwise, we need to check beyond the window
                left = i

                while right < n and s[right] == s[right - left]:
                    right += 1

                z[i] = right - left
                right -= 1

    return z


def longest_common_subsequence(s1: str, s2: str) -> str:
    """
    Find the longest common subsequence of two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The longest common subsequence
    """
    m, n = len(s1), len(s2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS
    i, j = m, n
    lcs = []

    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(lcs))


def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate the edit distance (Levenshtein distance) between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The minimum number of operations (insert, delete, replace) to transform s1 into s2
    """
    m, n = len(s1), len(s2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )

    return dp[m][n]


def longest_increasing_subsequence(nums: List[int]) -> List[int]:
    """
    Find the longest increasing subsequence in a list of numbers.

    Args:
        nums: List of numbers

    Returns:
        The longest increasing subsequence
    """
    if not nums:
        return []

    n = len(nums)

    # dp[i] = length of LIS ending at index i
    dp = [1] * n

    # prev[i] = previous index in the LIS ending at index i
    prev = [-1] * n

    # Fill DP table
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
                prev[i] = j

    # Find the index with maximum LIS length
    max_length = max(dp)
    max_index = dp.index(max_length)

    # Reconstruct the LIS
    lis = []
    while max_index != -1:
        lis.append(nums[max_index])
        max_index = prev[max_index]

    return list(reversed(lis))


def longest_increasing_subsequence_optimized(nums: List[int]) -> List[int]:
    """
    Find the longest increasing subsequence in a list of numbers using binary search.

    Args:
        nums: List of numbers

    Returns:
        The longest increasing subsequence
    """
    if not nums:
        return []

    n = len(nums)

    # tails[i] = smallest value that can end an increasing subsequence of length i+1
    tails = []

    # prev[i] = previous index in the LIS ending at index i
    prev = [-1] * n

    # Indices mapping tails positions to original indices
    indices = []

    for i, num in enumerate(nums):
        # Binary search to find the position to insert nums[i]
        pos = bisect.bisect_left(tails, num)

        if pos == len(tails):
            # Append to tails if num is greater than all elements
            tails.append(num)
            indices.append(i)
        else:
            # Replace the element at pos
            tails[pos] = num
            indices[pos] = i

        # Update prev array
        if pos > 0:
            prev[i] = indices[pos - 1]

    # Reconstruct the LIS
    lis = []
    curr = indices[-1]

    while curr != -1:
        lis.append(nums[curr])
        curr = prev[curr]

    return list(reversed(lis))


def manacher_algorithm(s: str) -> str:
    """
    Manacher's algorithm for finding the longest palindromic substring in linear time.

    Args:
        s: The input string

    Returns:
        The longest palindromic substring
    """
    if not s:
        return ""

    # Preprocess the string
    # Insert special character between each character and at boundaries
    # This handles both odd and even length palindromes
    t = '#' + '#'.join(s) + '#'
    n = len(t)

    # p[i] = radius of palindrome centered at i
    p = [0] * n

    center = 0  # Center of the rightmost palindrome
    right = 0   # Right boundary of the rightmost palindrome

    for i in range(n):
        # Initial value for p[i] using symmetry
        if right > i:
            p[i] = min(right - i, p[2 * center - i])

        # Expand palindrome centered at i
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1

        # Update center and right boundary if needed
        if i + p[i] > right:
            center = i
            right = i + p[i]

    # Find the maximum palindrome length
    max_len = max(p)
    center_index = p.index(max_len)

    # Convert back to original string indices
    start = (center_index - max_len) // 2
    end = start + max_len

    return s[start:end]


###########################################
# Dynamic Programming Algorithms
###########################################

def fibonacci_dp(n: int) -> int:
    """
    Calculate the nth Fibonacci number using dynamic programming.

    Args:
        n: The position in the Fibonacci sequence (0-indexed)

    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def coin_change(coins: List[int], amount: int) -> int:
    """
    Find the minimum number of coins needed to make up a given amount.

    Args:
        coins: List of coin denominations
        amount: Target amount

    Returns:
        Minimum number of coins needed, or -1 if impossible
    """
    # Initialize DP array with "infinity"
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins needed to make amount 0

    # Fill the DP array
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1


def knapsack_01(values: List[int], weights: List[int], capacity: int) -> int:
    """
    Solve the 0/1 Knapsack problem.

    Args:
        values: Values of the items
        weights: Weights of the items
        capacity: Maximum weight capacity of the knapsack

    Returns:
        Maximum value that can be obtained
    """
    n = len(values)

    # Create DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                # Either take the item or don't take it
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],  # Take the item
                    dp[i - 1][w]  # Don't take the item
                )
            else:
                # Can't take the item because it's too heavy
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]


def knapsack_01_reconstruct(values: List[int], weights: List[int], capacity: int) -> Tuple[int, List[int]]:
    """
    Solve the 0/1 Knapsack problem and reconstruct the solution.

    Args:
        values: Values of the items
        weights: Weights of the items
        capacity: Maximum weight capacity of the knapsack

    Returns:
        Tuple of (maximum value, list of selected item indices)
    """
    n = len(values)

    # Create DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                # Either take the item or don't take it
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],  # Take the item
                    dp[i - 1][w]  # Don't take the item
                )
            else:
                # Can't take the item because it's too heavy
                dp[i][w] = dp[i - 1][w]

    # Reconstruct the solution
    selected_items = []
    w = capacity

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            # Item i was selected
            selected_items.append(i - 1)
            w -= weights[i - 1]

    return dp[n][capacity], list(reversed(selected_items))


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Find the longest common substring of two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The longest common substring
    """
    m, n = len(s1), len(s2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Variables to keep track of the maximum length and ending position
    max_length = 0
    end_pos = 0

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0

    # Extract the longest common substring
    start = end_pos - max_length
    return s1[start:end_pos]


def matrix_chain_multiplication(dims: List[int]) -> int:
    """
    Solve the Matrix Chain Multiplication problem.

    Args:
        dims: Dimensions of matrices. For n matrices, dims has length n+1
              where dims[i-1] x dims[i] is the dimension of matrix i

    Returns:
        Minimum number of scalar multiplications needed
    """
    n = len(dims) - 1  # Number of matrices

    # Create DP table
    # dp[i][j] = minimum cost to multiply matrices i to j
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    # Fill the DP table
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            dp[i][j] = float('inf')

            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                dp[i][j] = min(dp[i][j], cost)

    return dp[1][n]


def max_subarray_sum(nums: List[int]) -> int:
    """
    Find the maximum sum of a contiguous subarray (Kadane's algorithm).

    Args:
        nums: List of integers

    Returns:
        Maximum sum of a contiguous subarray
    """
    if not nums:
        return 0

    current_sum = max_sum = nums[0]

    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum


def max_subarray_sum_indices(nums: List[int]) -> Tuple[int, int, int]:
    """
    Find the maximum sum of a contiguous subarray and its start/end indices.

    Args:
        nums: List of integers

    Returns:
        Tuple of (maximum sum, start index, end index)
    """
    if not nums:
        return 0, -1, -1

    current_sum = max_sum = nums[0]
    start = max_start = max_end = 0

    for i, num in enumerate(nums[1:], 1):
        if num > current_sum + num:
            current_sum = num
            start = i
        else:
            current_sum += num

        if current_sum > max_sum:
            max_sum = current_sum
            max_start = start
            max_end = i

    return max_sum, max_start, max_end


def rod_cutting(prices: List[int], n: int) -> int:
    """
    Solve the rod cutting problem.

    Args:
        prices: List of prices where prices[i] is the price of a rod of length i+1
        n: Length of the rod

    Returns:
        Maximum revenue that can be obtained
    """
    # Create DP table
    dp = [0] * (n + 1)

    # Fill the DP table
    for i in range(1, n + 1):
        max_val = float('-inf')
        for j in range(i):
            max_val = max(max_val, prices[j] + dp[i - j - 1])
        dp[i] = max_val

    return dp[n]


def rod_cutting_with_solution(prices: List[int], n: int) -> Tuple[int, List[int]]:
    """
    Solve the rod cutting problem and return the solution.

    Args:
        prices: List of prices where prices[i] is the price of a rod of length i+1
        n: Length of the rod

    Returns:
        Tuple of (maximum revenue, list of cut lengths)
    """
    # Create DP table and cut choice table
    dp = [0] * (n + 1)
    cut = [0] * (n + 1)

    # Fill the DP table
    for i in range(1, n + 1):
        max_val = float('-inf')
        for j in range(i):
            if max_val < prices[j] + dp[i - j - 1]:
                max_val = prices[j] + dp[i - j - 1]
                cut[i] = j + 1
        dp[i] = max_val

    # Reconstruct the solution
    result = []
    remaining = n

    while remaining > 0:
        result.append(cut[remaining])
        remaining -= cut[remaining]

    return dp[n], result


def longest_palindromic_subsequence(s: str) -> int:
    """
    Find the length of the longest palindromic subsequence in a string.

    Args:
        s: Input string

    Returns:
        Length of the longest palindromic subsequence
    """
    n = len(s)

    # Create DP table
    # dp[i][j] = length of LPS in s[i:j+1]
    dp = [[0] * n for _ in range(n)]

    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = 1

    # Fill the DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    return dp[0][n - 1]


def count_different_palindromic_subsequences(s: str) -> int:
    """
    Count different palindromic subsequences in a string.

    Args:
        s: Input string

    Returns:
        Number of different palindromic subsequences
    """
    n = len(s)
    MOD = 10**9 + 7

    # Create DP table
    # dp[i][j] = number of different palindromic subsequences in s[i:j+1]
    dp = [[0] * n for _ in range(n)]

    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = 1

    # Fill the DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] != s[j]:
                dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]) % MOD
            else:
                # Find the positions of the next and previous occurrence of s[i]
                left = i + 1
                right = j - 1

                while left <= right and s[left] != s[i]:
                    left += 1

                while left <= right and s[right] != s[i]:
                    right -= 1

                if left > right:
                    # No occurrence of s[i] in between
                    dp[i][j] = (2 * dp[i + 1][j - 1] + 2) % MOD
                elif left == right:
                    # One occurrence of s[i] in between
                    dp[i][j] = (2 * dp[i + 1][j - 1] + 1) % MOD
                else:
                    # More than one occurrence of s[i] in between
                    dp[i][j] = (2 * dp[i + 1][j - 1] - dp[left + 1][right - 1]) % MOD

    return (dp[0][n - 1] + MOD) % MOD  # Ensure positive result


def word_break(s: str, wordDict: List[str]) -> bool:
    """
    Determine if a string can be segmented into words from a dictionary.

    Args:
        s: Input string
        wordDict: List of dictionary words

    Returns:
        True if the string can be segmented, False otherwise
    """
    n = len(s)
    word_set = set(wordDict)

    # dp[i] = True if s[0:i] can be segmented
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string can be segmented

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]


def partition_equal_subset_sum(nums: List[int]) -> bool:
    """
    Determine if a list can be partitioned into two subsets with equal sum.

    Args:
        nums: List of integers

    Returns:
        True if the list can be partitioned, False otherwise
    """
    total_sum = sum(nums)

    # If the sum is odd, it cannot be partitioned
    if total_sum % 2 != 0:
        return False

    target = total_sum // 2
    n = len(nums)

    # dp[i][j] = True if a subset of nums[0..i-1] has sum j
    dp = [[False] * (target + 1) for _ in range(n + 1)]

    # Empty subset has sum 0
    for i in range(n + 1):
        dp[i][0] = True

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            if j < nums[i - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]

    return dp[n][target]


###########################################
# Searching and Sorting Algorithms
###########################################

def binary_search(arr: List[int], target: int) -> int:
    """
    Binary search algorithm for sorted arrays.

    Args:
        arr: Sorted list of integers
        target: The value to search for

    Returns:
        Index of the target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def binary_search_recursive(arr: List[int], target: int, left: int = 0, right: Optional[int] = None) -> int:
    """
    Recursive binary search algorithm for sorted arrays.

    Args:
        arr: Sorted list of integers
        target: The value to search for
        left: Left boundary index
        right: Right boundary index

    Returns:
        Index of the target if found, -1 otherwise
    """
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


def lower_bound(arr: List[int], target: int) -> int:
    """
    Find the first position where target could be inserted without changing the order.

    Args:
        arr: Sorted list of integers
        target: The value to search for

    Returns:
        Index of the first element not less than target
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left


def upper_bound(arr: List[int], target: int) -> int:
    """
    Find the first position where target+1 could be inserted without changing the order.

    Args:
        arr: Sorted list of integers
        target: The value to search for

    Returns:
        Index of the first element greater than target
    """
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid

    return left


def quick_sort(arr: List[int]) -> List[int]:
    """
    Quick sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: List[int]) -> List[int]:
    """
    Merge sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """Helper function for merge sort."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


def heap_sort(arr: List[int]) -> List[int]:
    """
    Heap sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list
    """
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr


def counting_sort(arr: List[int], max_val: Optional[int] = None) -> List[int]:
    """
    Counting sort algorithm for non-negative integers.

    Args:
        arr: List of non-negative integers to sort
        max_val: Maximum value in the array (calculated if not provided)

    Returns:
        Sorted list
    """
    if not arr:
        return []

    if max_val is None:
        max_val = max(arr)

    # Create count array
    count = [0] * (max_val + 1)

    # Count occurrences
    for num in arr:
        count[num] += 1

    # Reconstruct the sorted array
    sorted_arr = []
    for i in range(max_val + 1):
        sorted_arr.extend([i] * count[i])

    return sorted_arr


def radix_sort(arr: List[int]) -> List[int]:
    """
    Radix sort algorithm for non-negative integers.

    Args:
        arr: List of non-negative integers to sort

    Returns:
        Sorted list
    """
    if not arr:
        return []

    # Find the maximum number to know the number of digits
    max_val = max(arr)

    # Do counting sort for every digit
    exp = 1
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr


def counting_sort_by_digit(arr: List[int], exp: int) -> List[int]:
    """Helper function for radix sort."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    # Count occurrences of each digit
    for num in arr:
        index = (num // exp) % 10
        count[index] += 1

    # Change count[i] to contain actual position of this digit in output
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    return output


###########################################
# Geometric Algorithms
###########################################

@dataclass
class Point:
    """Point in 2D space."""
    x: float
    y: float


def orientation(p: Point, q: Point, r: Point) -> int:
    """
    Determine the orientation of three points.

    Args:
        p, q, r: Three points

    Returns:
        0 if collinear, 1 if clockwise, 2 if counterclockwise
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise


def convex_hull(points: List[Point]) -> List[Point]:
    """
    Compute the convex hull of a set of points using Graham's scan algorithm.

    Args:
        points: List of points

    Returns:
        List of points forming the convex hull
    """
    n = len(points)

    # Need at least 3 points for a convex hull
    if n < 3:
        return points

    # Find the bottom-most point (and left-most in case of tie)
    bottom_idx = 0
    for i in range(1, n):
        if points[i].y < points[bottom_idx].y or (points[i].y == points[bottom_idx].y and points[i].x < points[bottom_idx].x):
            bottom_idx = i

    # Swap the bottom-most point with the first point
    points[0], points[bottom_idx] = points[bottom_idx], points[0]

    # Sort points by polar angle with respect to the bottom-most point
    p0 = points[0]

    def compare(p1, p2):
        o = orientation(p0, p1, p2)

        if o == 0:
            # Collinear points, sort by distance from p0
            dist1 = (p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2
            dist2 = (p2.x - p0.x) ** 2 + (p2.y - p0.y) ** 2
            return -1 if dist1 < dist2 else 1

        return -1 if o == 2 else 1

    points[1:] = sorted(points[1:], key=functools.cmp_to_key(compare))

    # Remove collinear points with the first point (keep the farthest)
    m = 1
    for i in range(1, n):
        while i < n - 1 and orientation(p0, points[i], points[i + 1]) == 0:
            i += 1
        points[m] = points[i]
        m += 1

    # If we have less than 3 points, convex hull is not possible
    if m < 3:
        return points[:m]

    # Build the convex hull
    hull = [points[0], points[1], points[2]]

    for i in range(3, m):
        # Remove points that make a non-left turn
        while len(hull) > 1 and orientation(hull[-2], hull[-1], points[i]) != 2:
            hull.pop()

        hull.append(points[i])

    return hull


def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """
    Check if a point is inside a polygon using the ray casting algorithm.

    Args:
        point: The point to check
        polygon: List of points forming the polygon

    Returns:
        True if the point is inside the polygon, False otherwise
    """
    n = len(polygon)

    if n < 3:
        return False

    # Check if the point is on an edge
    for i in range(n):
        j = (i + 1) % n

        if (polygon[i].y == polygon[j].y and
            polygon[i].y == point.y and
            point.x >= min(polygon[i].x, polygon[j].x) and
            point.x <= max(polygon[i].x, polygon[j].x)):
            return True

        if (polygon[i].x == polygon[j].x and
            polygon[i].x == point.x and
            point.y >= min(polygon[i].y, polygon[j].y) and
            point.y <= max(polygon[i].y, polygon[j].y)):
            return True

    # Ray casting algorithm
    inside = False

    for i in range(n):
        j = (i + 1) % n

        if ((polygon[i].y > point.y) != (polygon[j].y > point.y) and
            point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x):
            inside = not inside

    return inside


def closest_pair_of_points(points: List[Point]) -> Tuple[Point, Point, float]:
    """
    Find the closest pair of points in a set of points.

    Args:
        points: List of points

    Returns:
        Tuple of (point1, point2, min_distance)
    """
    def distance(p1, p2):
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

    def closest_pair_recursive(points_x, points_y):
        n = len(points_x)

        # Base case: if we have 2 or 3 points, compute minimum distance directly
        if n <= 3:
            min_dist = float('inf')
            min_pair = (None, None)

            for i in range(n):
                for j in range(i + 1, n):
                    dist = distance(points_x[i], points_x[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (points_x[i], points_x[j])

            return min_pair[0], min_pair[1], min_dist

        # Divide points into two halves
        mid = n // 2
        mid_point = points_x[mid]

        # Divide the points into left and right halves
        points_x_left = points_x[:mid]
        points_x_right = points_x[mid:]

        # Divide points_y into left and right halves
        points_y_left = []
        points_y_right = []

        for p in points_y:
            if p.x <= mid_point.x:
                points_y_left.append(p)
            else:
                points_y_right.append(p)

        # Recursively find the closest pair in left and right halves
        p1_left, p2_left, dist_left = closest_pair_recursive(points_x_left, points_y_left)
        p1_right, p2_right, dist_right = closest_pair_recursive(points_x_right, points_y_right)

        # Find the smaller distance
        if dist_left < dist_right:
            p1, p2, min_dist = p1_left, p2_left, dist_left
        else:
            p1, p2, min_dist = p1_right, p2_right, dist_right

        # Build a strip of points whose x-coordinates are within min_dist of mid_point
        strip = []
        for p in points_y:
            if abs(p.x - mid_point.x) < min_dist:
                strip.append(p)

        # Find the closest pair within the strip
        for i in range(len(strip)):
            # Check at most 7 points ahead
            j = i + 1
            while j < len(strip) and strip[j].y - strip[i].y < min_dist:
                dist = distance(strip[i], strip[j])
                if dist < min_dist:
                    min_dist = dist
                    p1, p2 = strip[i], strip[j]
                j += 1

        return p1, p2, min_dist

    # Sort points by x-coordinate and y-coordinate
    points_x = sorted(points, key=lambda p: p.x)
    points_y = sorted(points, key=lambda p: p.y)

    return closest_pair_recursive(points_x, points_y)


def line_segment_intersection(p1: Point, q1: Point, p2: Point, q2: Point) -> Optional[Point]:
    """
    Find the intersection point of two line segments.

    Args:
        p1, q1: First line segment from p1 to q1
        p2, q2: Second line segment from p2 to q2

    Returns:
        Intersection point if the line segments intersect, None otherwise
    """
    def on_segment(p, q, r):
        return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))

    def orientation(p, q, r):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    # Find the four orientations
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        # Calculate the intersection point
        a1 = q1.y - p1.y
        b1 = p1.x - q1.x
        c1 = a1 * p1.x + b1 * p1.y

        a2 = q2.y - p2.y
        b2 = p2.x - q2.x
        c2 = a2 * p2.x + b2 * p2.y

        determinant = a1 * b2 - a2 * b1

        if determinant == 0:
            return None  # Parallel lines

        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant

        return Point(x, y)

    # Special cases
    if o1 == 0 and on_segment(p1, p2, q1):
        return p2
    if o2 == 0 and on_segment(p1, q2, q1):
        return q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return p1
    if o4 == 0 and on_segment(p2, q1, q2):
        return q1

    return None  # No intersection


###########################################
# Utility and Visualization Functions
###########################################

def visualize_graph(graph: Graph, layout='spring', node_labels=True, edge_labels=True):
    """
    Visualize a graph using matplotlib.

    Args:
        graph: The graph to visualize
        layout: The layout algorithm to use ('spring', 'circular', etc.)
        node_labels: Whether to show node labels
        edge_labels: Whether to show edge labels
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        # Convert our graph to NetworkX graph
        G = nx.DiGraph() if graph.directed else nx.Graph()

        # Add vertices
        for vertex in graph.vertices:
            G.add_node(vertex)

        # Add edges
        for (u, v), weight in graph.edges.items():
            weight_val = weight if graph.weighted else None
            G.add_edge(u, v, weight=weight_val)

        # Create the layout
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Draw the graph
        plt.figure(figsize=(10, 8))

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, arrowsize=20, width=1.5)

        # Draw labels if requested
        if node_labels:
            nx.draw_networkx_labels(G, pos, font_size=12)

        if edge_labels and graph.weighted:
            edge_labels_dict = {(u, v): f"{w}" for (u, v), w in graph.edges.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_size=10)

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("NetworkX and matplotlib are required for graph visualization.")


def visualize_binary_tree(root: BinaryTreeNode):
    """
    Visualize a binary tree using matplotlib.

    Args:
        root: The root node of the binary tree
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        def build_graph(node, graph, pos, x=0, y=0, layer=1):
            if node is None:
                return

            graph.add_node(id(node), value=node.value)
            pos[id(node)] = (x, y)

            if node.left:
                graph.add_node(id(node.left), value=node.left.value)
                graph.add_edge(id(node), id(node.left))
                build_graph(node.left, graph, pos, x - 1/2**layer, y - 1, layer + 1)

            if node.right:
                graph.add_node(id(node.right), value=node.right.value)
                graph.add_edge(id(node), id(node.right))
                build_graph(node.right, graph, pos, x + 1/2**layer, y - 1, layer + 1)

        G = nx.DiGraph()
        pos = {}
        build_graph(root, G, pos)

        plt.figure(figsize=(10, 8))

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, arrows=False)

        # Draw labels
        labels = {node: G.nodes[node]['value'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("NetworkX and matplotlib are required for tree visualization.")


def visualize_sorting_algorithm(algorithm: Callable, arr: List[int], title: str = None):
    """
    Visualize a sorting algorithm using an animation.

    Args:
        algorithm: The sorting algorithm function
        arr: The array to sort
        title: The title of the animation
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        # Copy the array to avoid modifying the original
        arr = arr.copy()

        # Get all steps in the sorting process
        history = []

        def sorting_algorithm_with_history(arr):
            nonlocal history
            history.append(arr.copy())
            result = algorithm(arr)
            history.append(arr.copy() if result is None else result.copy())
            return result

        sorting_algorithm_with_history(arr)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        if title:
            ax.set_title(title)

        bars = ax.bar(range(len(arr)), history[0], color='skyblue')
        ax.set_xlim(-0.5, len(arr) - 0.5)
        ax.set_ylim(0, max(arr) * 1.1)

        # Update function for animation
        def update(frame):
            for i, val in enumerate(history[frame]):
                bars[i].set_height(val)
            return bars

        # Create animation
        anim = FuncAnimation(fig, update, frames=len(history), interval=100, repeat=False)

        plt.tight_layout()
        plt.show()

        return anim

    except ImportError:
        print("Matplotlib is required for sorting visualization.")


def visualize_convex_hull(points: List[Point], hull_points: List[Point] = None):
    """
    Visualize the convex hull of a set of points.

    Args:
        points: List of points
        hull_points: List of points forming the convex hull (computed if not provided)
    """
    try:
        import matplotlib.pyplot as plt

        # Compute convex hull if not provided
        if hull_points is None:
            hull_points = convex_hull(points)

        # Extract x and y coordinates
        x = [p.x for p in points]
        y = [p.y for p in points]

        hull_x = [p.x for p in hull_points]
        hull_y = [p.y for p in hull_points]

        # Add the first point again to close the polygon
        hull_x.append(hull_points[0].x)
        hull_y.append(hull_points[0].y)

        # Create the figure
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, color='blue', label='Points')
        plt.plot(hull_x, hull_y, 'r-', linewidth=2, label='Convex Hull')
        plt.scatter(hull_x[:-1], hull_y[:-1], color='red')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.title('Convex Hull')
        plt.show()

    except ImportError:
        print("Matplotlib is required for convex hull visualization.")


def measure_execution_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure the execution time of a function.

    Args:
        func: The function to measure
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Tuple of (function_result, execution_time_in_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    return result, end_time - start_time


def stress_test(func1: Callable, func2: Callable, input_generator: Callable, n_tests: int = 100) -> bool:
    """
    Stress test two implementations of the same algorithm against each other.

    Args:
        func1: First implementation
        func2: Second implementation
        input_generator: Function that generates random input
        n_tests: Number of tests to run

    Returns:
        True if all tests pass, False otherwise
    """
    for i in range(n_tests):
        input_data = input_generator()

        try:
            result1 = func1(input_data)
            result2 = func2(input_data)

            if result1 != result2:
                print(f"Test failed on input: {input_data}")
                print(f"func1 output: {result1}")
                print(f"func2 output: {result2}")
                return False

        except Exception as e:
            print(f"Exception on test {i} with input: {input_data}")
            print(f"Exception: {e}")
            return False

    print(f"All {n_tests} tests passed!")
    return True


###########################################
# Main Examples
###########################################

def graph_algorithms_example():
    """Example usage of graph algorithms."""
    # Create a directed, weighted graph
    g = Graph(directed=True, weighted=True)

    # Add vertices
    for i in range(6):
        g.add_vertex(i)

    # Add edges
    g.add_edge(0, 1, 5)
    g.add_edge(0, 2, 3)
    g.add_edge(1, 3, 6)
    g.add_edge(1, 2, 2)
    g.add_edge(2, 4, 4)
    g.add_edge(2, 3, 7)
    g.add_edge(3, 5, 1)
    g.add_edge(4, 5, 2)

    print("Graph:")
    print(g)

    # Run BFS
    print("\nBFS from vertex 0:")
    bfs_result = breadth_first_search(g, 0)
    for vertex, parent in bfs_result.items():
        if parent is not None:
            print(f"Vertex {vertex}: parent = {parent}")
        else:
            print(f"Vertex {vertex}: root")

    # Run DFS
    print("\nDFS from vertex 0:")
    dfs_result = depth_first_search(g, 0)
    for vertex, (discovery, finish, parent) in dfs_result.items():
        if parent is not None:
            print(f"Vertex {vertex}: parent = {parent}, discovery = {discovery}, finish = {finish}")
        else:
            print(f"Vertex {vertex}: root, discovery = {discovery}, finish = {finish}")

    # Run Dijkstra's algorithm
    print("\nDijkstra's algorithm from vertex 0:")
    distances, predecessors = dijkstra(g, 0)
    for vertex, distance in distances.items():
        path = reconstruct_shortest_path(0, vertex, predecessors)
        print(f"Vertex {vertex}: distance = {distance}, path = {path}")

    # Run topological sort
    print("\nTopological sort:")
    topo_order = topological_sort(g)
    print(topo_order)

    # Visualize the graph
    print("\nVisualizing the graph...")
    visualize_graph(g)


def dynamic_programming_example():
    """Example usage of dynamic programming algorithms."""
    # Knapsack problem
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    print("Knapsack Problem:")
    max_value, selected_items = knapsack_01_with_solution(values, weights, capacity)
    print(f"Maximum value: {max_value}")
    print(f"Selected items: {selected_items}")

    # Longest Common Subsequence
    s1 = "ABCBDAB"
    s2 = "BDCABA"

    print("\nLongest Common Subsequence:")
    lcs = longest_common_subsequence(s1, s2)
    print(f"LCS of '{s1}' and '{s2}': '{lcs}'")

    # Longest Increasing Subsequence
    nums = [10, 22, 9, 33, 21, 50, 41, 60, 80]

    print("\nLongest Increasing Subsequence:")
    lis = longest_increasing_subsequence(nums)
    print(f"LIS of {nums}: {lis}")

    # Edit Distance
    s1 = "kitten"
    s2 = "sitting"

    print("\nEdit Distance:")
    distance = edit_distance(s1, s2)
    print(f"Edit distance between '{s1}' and '{s2}': {distance}")

    # Rod Cutting
    prices = [1, 5, 8, 9, 10, 17, 17, 20]
    rod_length = 8

    print("\nRod Cutting Problem:")
    max_revenue, cuts = rod_cutting_with_solution(prices, rod_length)
    print(f"Maximum revenue: {max_revenue}")
    print(f"Optimal cuts: {cuts}")


def string_algorithms_example():
    """Example usage of string algorithms."""
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"

    print("String Pattern Matching:")

    # Knuth-Morris-Pratt (KMP) algorithm
    print("\nKMP Algorithm:")
    kmp_result = kmp_search(text, pattern)
    print(f"Pattern '{pattern}' found at positions: {kmp_result}")

    # Rabin-Karp algorithm
    print("\nRabin-Karp Algorithm:")
    rk_result = rabin_karp_search(text, pattern)
    print(f"Pattern '{pattern}' found at positions: {rk_result}")

    # Z algorithm
    print("\nZ Algorithm:")
    z_array = z_algorithm(pattern + "$" + text)
    print(f"Z array: {z_array}")
    pattern_matches = [i - len(pattern) - 1 for i in range(len(pattern) + 1, len(z_array)) if z_array[i] == len(pattern)]
    print(f"Pattern '{pattern}' found at positions: {pattern_matches}")

    # Longest Palindromic Substring
    s = "babad"

    print("\nLongest Palindromic Substring:")
    lps = longest_palindromic_substring(s)
    print(f"Longest palindromic substring of '{s}': '{lps}'")

    # Manacher's Algorithm
    print("\nManacher's Algorithm:")
    lps_manacher = manacher_algorithm(s)
    print(f"Longest palindromic substring using Manacher's algorithm: '{lps_manacher}'")


def geometric_algorithms_example():
    """Example usage of geometric algorithms."""
    # Create random points
    import random
    random.seed(42)

    points = [Point(random.randint(0, 100), random.randint(0, 100)) for _ in range(15)]

    print("Geometric Algorithms:")

    # Convex Hull
    print("\nConvex Hull:")
    hull = convex_hull(points)
    print(f"Convex hull contains {len(hull)} points out of {len(points)} total points")

    # Visualize Convex Hull
    print("\nVisualizing Convex Hull...")
    visualize_convex_hull(points, hull)

    # Closest Pair of Points
    print("\nClosest Pair of Points:")
    p1, p2, min_dist = closest_pair_of_points(points)
    print(f"Closest points: ({p1.x}, {p1.y}) and ({p2.x}, {p2.y})")
    print(f"Distance: {min_dist}")

    # Line Segment Intersection
    print("\nLine Segment Intersection:")
    line1 = (Point(10, 10), Point(50, 50))
    line2 = (Point(10, 50), Point(50, 10))

    intersection = line_segment_intersection(line1[0], line1[1], line2[0], line2[1])

    if intersection:
        print(f"Lines intersect at point ({intersection.x}, {intersection.y})")
    else:
        print("Lines do not intersect")


def sorting_algorithms_example():
    """Example usage of sorting algorithms."""
    # Create a random array
    import random
    random.seed(42)

    arr = [random.randint(1, 100) for _ in range(20)]

    print("Sorting Algorithms:")
    print(f"Original array: {arr}")

    # Quick Sort
    print("\nQuick Sort:")
    sorted_arr = quick_sort(arr.copy())
    print(f"Sorted array: {sorted_arr}")

    # Merge Sort
    print("\nMerge Sort:")
    sorted_arr = merge_sort(arr.copy())
    print(f"Sorted array: {sorted_arr}")

    # Heap Sort
    print("\nHeap Sort:")
    sorted_arr = heap_sort(arr.copy())
    print(f"Sorted array: {sorted_arr}")

    # Visualize Quick Sort
    print("\nVisualizing Quick Sort...")
    visualize_sorting_algorithm(quick_sort, arr, "Quick Sort")


def main():
    """Main function to demonstrate the usage of algorithms."""
    print("Advanced Algorithms and Data Structures Toolkit")
    print("==============================================")

    examples = {
        1: ("Graph Algorithms", graph_algorithms_example),
        2: ("Dynamic Programming", dynamic_programming_example),
        3: ("String Algorithms", string_algorithms_example),
        4: ("Geometric Algorithms", geometric_algorithms_example),
        5: ("Sorting Algorithms", sorting_algorithms_example)
    }

    print("\nAvailable Examples:")
    for num, (name, _) in examples.items():
        print(f"{num}. {name}")

    try:
        choice = int(input("\nEnter the number of the example to run (or 0 to run all): "))

        if choice == 0:
            for _, (name, func) in examples.items():
                print(f"\n=== Running {name} Example ===")
                func()
                print("\nPress Enter to continue...")
                input()
        elif choice in examples:
            name, func = examples[choice]
            print(f"\n=== Running {name} Example ===")
            func()
        else:
            print("Invalid choice!")

    except ValueError:
        print("Please enter a valid number!")


if __name__ == "__main__":
    main()