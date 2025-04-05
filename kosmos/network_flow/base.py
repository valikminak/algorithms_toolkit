# kosmos/flow/base.py
from collections import defaultdict, deque
from typing import List, Any


class FlowNetwork:
    """
    Flow network representation for network flow algorithms.
    """

    def __init__(self):
        self.graph = defaultdict(dict)
        self.vertices = set()

    def add_edge(self, u: Any, v: Any, capacity: int) -> None:
        """Add an edge to the flow network"""
        self.graph[u][v] = capacity
        # Add reverse edge with 0 capacity if it doesn't exist
        if v not in self.graph or u not in self.graph[v]:
            self.graph[v][u] = 0

        self.vertices.add(u)
        self.vertices.add(v)

    def get_neighbors(self, u: Any) -> List[Any]:
        """Get all neighbors of a vertex"""
        return list(self.graph[u].keys())

    def get_capacity(self, u: Any, v: Any) -> int:
        """Get capacity of an edge"""
        return self.graph[u].get(v, 0)

    def set_flow(self, u: Any, v: Any, flow: int) -> None:
        """Set flow on an edge"""
        self.graph[u][v] = flow


class FordFulkerson:
    """
    Ford-Fulkerson algorithm for maximum flow problem.
    Uses depth-first search to find augmenting paths.
    """

    @staticmethod
    def max_flow(graph: FlowNetwork, source: Any, sink: Any):
        """
        Find maximum flow from source to sink in a flow network.

        Args:
            graph: Flow network
            source: Source vertex
            sink: Sink vertex

        Returns:
            Maximum flow value
        """
        # Create residual graph
        residual = FlowNetwork()
        for u in graph.vertices:
            for v, capacity in graph.graph[u].items():
                residual.add_edge(u, v, capacity)

        max_flow = 0
        path_flows = []
        paths = []

        # Find augmenting path using DFS
        def dfs(u, flow, visited):
            if u == sink:
                return flow

            visited.add(u)

            for v in residual.get_neighbors(u):
                capacity = residual.get_capacity(u, v)
                if v not in visited and capacity > 0:
                    min_flow = min(flow, capacity)
                    bottleneck = dfs(v, min_flow, visited)

                    if bottleneck > 0:
                        # Update residual capacities
                        residual.graph[u][v] -= bottleneck
                        residual.graph[v][u] += bottleneck
                        return bottleneck

            return 0

        # Find augmenting paths until no more exist
        while True:
            visited = set()
            path_flow = dfs(source, float('inf'), visited)

            if path_flow == 0:
                break

            max_flow += path_flow
            path_flows.append(path_flow)

            # Record the path for visualization
            path = []
            u = sink
            while u != source:
                for v in residual.vertices:
                    if v in residual.graph and u in residual.graph[v] and residual.graph[v][u] > 0:
                        path.append(u)
                        u = v
                        break
            path.append(source)
            paths.append(list(reversed(path)))

        return max_flow, path_flows, paths


class EdmondsKarp:
    """
    Edmonds-Karp algorithm for maximum flow problem.
    Uses breadth-first search to find augmenting paths.
    """

    @staticmethod
    def max_flow(graph: FlowNetwork, source: Any, sink: Any):
        """
        Find maximum flow from source to sink in a flow network.

        Args:
            graph: Flow network
            source: Source vertex
            sink: Sink vertex

        Returns:
            Maximum flow value
        """
        # Create residual graph
        residual = FlowNetwork()
        for u in graph.vertices:
            for v, capacity in graph.graph[u].items():
                residual.add_edge(u, v, capacity)

        max_flow = 0
        path_flows = []
        paths = []

        # Find augmenting path using BFS
        def bfs():
            visited = {source: None}
            queue = deque([source])

            while queue:
                u = queue.popleft()

                if u == sink:
                    break

                for v in residual.get_neighbors(u):
                    capacity = residual.get_capacity(u, v)
                    if v not in visited and capacity > 0:
                        queue.append(v)
                        visited[v] = u

            # Reconstruct path
            if sink in visited:
                path = []
                u = sink
                while u != source:
                    path.append(u)
                    u = visited[u]
                path.append(source)
                path.reverse()

                # Find bottleneck capacity
                flow = float('inf')
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    flow = min(flow, residual.get_capacity(u, v))

                return path, flow

            return None, 0

        # Find augmenting paths until no more exist
        while True:
            path, flow = bfs()

            if flow == 0:
                break

            paths.append(path)
            path_flows.append(flow)
            max_flow += flow

            # Update residual capacities
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                residual.graph[u][v] -= flow
                residual.graph[v][u] += flow

        return max_flow, path_flows, paths