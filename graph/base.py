from typing import List, Dict, Tuple, Set, Optional, Any, Callable, Union


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
        self.adj_list = {}  # Dictionary mapping vertices to adjacent vertices

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
        if self.weighted:
            self.edges[(u, v)] = weight if weight is not None else 1.0
        else:
            self.edges[(u, v)] = None

        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)

        # If undirected, add the reverse edge
        if not self.directed:
            if self.weighted:
                self.edges[(v, u)] = weight if weight is not None else 1.0
            else:
                self.edges[(v, u)] = None

            if u not in self.adj_list[v]:
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
                if vertex in self.adj_list[v]:
                    self.adj_list[v].remove(vertex)

    def get_neighbors(self, vertex: Any) -> List[Any]:
        """Get all neighbors of a vertex."""
        return self.adj_list.get(vertex, [])

    def get_edge_weight(self, u: Any, v: Any) -> Optional[float]:
        """Get the weight of an edge."""
        return self.edges.get((u, v))

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
        result += f"{len(self.vertices)} vertices and {len(self.edges) // (1 if self.directed else 2)} edges.\n"

        for vertex in sorted(self.vertices, key=str):
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
        result = {}
        for x in self.parent:
            root = self.find(x)
            if root not in result:
                result[root] = []
            result[root].append(x)
        return result