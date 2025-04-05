# kosmos/flow/min_cut.py
from typing import Any, List, Tuple
from collections import deque
from kosmos.network_flow.base import FlowNetwork, FordFulkerson


class MinCut:
    """
    Minimum Cut algorithm using Ford-Fulkerson method.
    """

    @staticmethod
    def min_cut(graph: FlowNetwork, source: Any, sink: Any) -> Tuple[int, List[Tuple[Any, Any]]]:
        """
        Find the minimum cut in a flow network.

        Args:
            graph: Flow network
            source: Source vertex
            sink: Sink vertex

        Returns:
            Tuple of (min cut value, edges in the cut)
        """
        # First find max flow using Ford-Fulkerson
        max_flow, _, _ = FordFulkerson.max_flow(graph, source, sink)

        # Create residual graph after max flow
        residual = FlowNetwork()
        for u in graph.vertices:
            for v, capacity in graph.graph[u].items():
                residual.add_edge(u, v, capacity)

        # Find vertices reachable from source in residual network
        reachable = set()
        queue = deque([source])
        reachable.add(source)

        while queue:
            u = queue.popleft()
            for v in residual.get_neighbors(u):
                if residual.get_capacity(u, v) > 0 and v not in reachable:
                    reachable.add(v)
                    queue.append(v)

        # The min cut consists of edges from reachable to non-reachable vertices
        cut_edges = []
        for u in reachable:
            for v in graph.graph[u]:
                if v not in reachable:
                    cut_edges.append((u, v))

        return max_flow, cut_edges, reachable, set(graph.vertices) - reachable