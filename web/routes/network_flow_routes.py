from flask import Blueprint, jsonify, request
from collections import defaultdict

from kosmos.network_flow.base import FlowNetwork, FordFulkerson, EdmondsKarp
from kosmos.network_flow.min_cut import MinCut

network_flow_routes = Blueprint('network_flow_routes', __name__)


@network_flow_routes.route('/api/flow/algorithms')
def get_flow_algorithms():
    """Return available network flow algorithms"""
    algorithms = [
        {"id": "ford_fulkerson", "name": "Ford-Fulkerson", "complexity": "O(E·max_flow)"},
        {"id": "edmonds_karp", "name": "Edmonds-Karp", "complexity": "O(V·E²)"},
        {"id": "min_cut", "name": "Minimum Cut", "complexity": "O(V·E²)"}
    ]
    return jsonify(algorithms)


@network_flow_routes.route('/api/flow/ford_fulkerson', methods=['POST'])
def ford_fulkerson_route():
    """
    Ford-Fulkerson algorithm implementation
    Input: JSON with graph adjacency list, capacities, source, and sink
    Output: JSON with max flow, flow graph, and steps
    """
    try:
        data = request.get_json()
        graph_data = data.get('graph', {})
        source = data.get('source')
        sink = data.get('sink')

        if not graph_data:
            return jsonify({'error': 'Graph data is required'}), 400
        if source is None or sink is None:
            return jsonify({'error': 'Source and sink nodes are required'}), 400

        # Create flow network
        flow_network = FlowNetwork()
        for u, neighbors in graph_data.items():
            for v, capacity in neighbors.items():
                flow_network.add_edge(u, v, capacity)

        # Run Ford-Fulkerson algorithm
        max_flow, path_flows, paths = FordFulkerson.max_flow(flow_network, source, sink)

        # Generate steps for visualization
        steps = []
        residual_graph = FlowNetwork()
        flow_graph = defaultdict(dict)

        # Initialize residual graph with original capacities
        for u in flow_network.vertices:
            for v, capacity in flow_network.graph[u].items():
                residual_graph.add_edge(u, v, capacity)
                flow_graph[u][v] = 0

        # Replay the algorithm steps
        for i, (path, flow) in enumerate(zip(paths, path_flows)):
            # Update residual capacities and flow
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                residual_graph.graph[u][v] -= flow
                residual_graph.graph[v][u] += flow
                flow_graph[u][v] += flow

            # Record the step
            steps.append({
                'path': path,
                'path_flow': flow,
                'max_flow_so_far': sum(path_flows[:i + 1]),
                'residual_graph': {u: dict(residual_graph.graph[u]) for u in residual_graph.vertices},
                'flow_graph': {u: dict(flow_graph[u]) for u in flow_graph},
                'info': f"Found augmenting path: {' → '.join(path)} with flow {flow}"
            })

        # Convert flow graph to a cleaner format
        final_flow = {}
        for u in flow_network.vertices:
            final_flow[u] = {}
            for v in flow_network.graph[u]:
                if flow_graph[u][v] > 0:
                    final_flow[u][v] = flow_graph[u][v]

        return jsonify({
            'algorithm': 'ford_fulkerson',
            'category': 'flow',
            'source': source,
            'sink': sink,
            'max_flow': max_flow,
            'flow_graph': final_flow,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@network_flow_routes.route('/api/flow/edmonds_karp', methods=['POST'])
def edmonds_karp_route():
    """
    Edmonds-Karp algorithm implementation
    Input: JSON with graph adjacency list, capacities, source, and sink
    Output: JSON with max flow, flow graph, and steps
    """
    try:
        data = request.get_json()
        graph_data = data.get('graph', {})
        source = data.get('source')
        sink = data.get('sink')

        if not graph_data:
            return jsonify({'error': 'Graph data is required'}), 400
        if source is None or sink is None:
            return jsonify({'error': 'Source and sink nodes are required'}), 400

        # Create flow network
        flow_network = FlowNetwork()
        for u, neighbors in graph_data.items():
            for v, capacity in neighbors.items():
                flow_network.add_edge(u, v, capacity)

        # Run Edmonds-Karp algorithm
        max_flow, path_flows, paths = EdmondsKarp.max_flow(flow_network, source, sink)

        # Generate steps for visualization
        steps = []
        residual_graph = FlowNetwork()
        flow_graph = defaultdict(dict)

        # Initialize residual graph with original capacities
        for u in flow_network.vertices:
            for v, capacity in flow_network.graph[u].items():
                residual_graph.add_edge(u, v, capacity)
                flow_graph[u][v] = 0

        # Replay the algorithm steps
        for i, (path, flow) in enumerate(zip(paths, path_flows)):
            # Record BFS steps (simplified)
            bfs_steps = []
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                bfs_steps.append({
                    'node': v,
                    'from': u,
                    'capacity': residual_graph.graph[u][v]
                })

            # Update residual capacities and flow
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                residual_graph.graph[u][v] -= flow
                residual_graph.graph[v][u] += flow
                flow_graph[u][v] += flow

            # Record the step
            steps.append({
                'iteration': i + 1,
                'path': path,
                'path_flow': flow,
                'max_flow_so_far': sum(path_flows[:i + 1]),
                'bfs_steps': bfs_steps,
                'residual_graph': {u: dict(residual_graph.graph[u]) for u in residual_graph.vertices},
                'flow_graph': {u: dict(flow_graph[u]) for u in flow_graph},
                'info': f"BFS found augmenting path: {' → '.join(path)} with flow {flow}"
            })

        # Convert flow graph to a cleaner format
        final_flow = {}
        for u in flow_network.vertices:
            final_flow[u] = {}
            for v in flow_network.graph[u]:
                if flow_graph[u][v] > 0:
                    final_flow[u][v] = flow_graph[u][v]

        return jsonify({
            'algorithm': 'edmonds_karp',
            'category': 'flow',
            'source': source,
            'sink': sink,
            'max_flow': max_flow,
            'flow_graph': final_flow,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@network_flow_routes.route('/api/flow/min_cut', methods=['POST'])
def min_cut_route():
    """
    Minimum Cut algorithm using Ford-Fulkerson
    Input: JSON with graph adjacency list, capacities, source, and sink
    Output: JSON with min cut value, cut edges, and partitions
    """
    try:
        data = request.get_json()
        graph_data = data.get('graph', {})
        source = data.get('source')
        sink = data.get('sink')

        if not graph_data:
            return jsonify({'error': 'Graph data is required'}), 400
        if source is None or sink is None:
            return jsonify({'error': 'Source and sink nodes are required'}), 400

        # Create flow network
        flow_network = FlowNetwork()
        for u, neighbors in graph_data.items():
            for v, capacity in neighbors.items():
                flow_network.add_edge(u, v, capacity)

        # Run MinCut algorithm
        min_cut_value, cut_edges, s_partition, t_partition = MinCut.min_cut(flow_network, source, sink)

        # Generate steps for visualization
        # First run max flow steps
        max_flow, path_flows, paths = FordFulkerson.max_flow(flow_network, source, sink)

        # Create steps for min cut
        steps = []

        # Initial step - show network
        steps.append({
            'phase': 'init',
            'info': 'Initial flow network',
            'source': source,
            'sink': sink
        })

        # Max flow computation steps (simplified for min cut visualization)
        for i, (path, flow) in enumerate(zip(paths, path_flows)):
            steps.append({
                'phase': 'max_flow',
                'path': path,
                'flow': flow,
                'max_flow_so_far': sum(path_flows[:i + 1]),
                'info': f"Computing max flow: Found path with flow {flow}"
            })

        # Min cut step - show partitioning
        steps.append({
            'phase': 'partition',
            'min_cut': min_cut_value,
            'cut_edges': [{'from': u, 'to': v} for u, v in cut_edges],
            's_partition': list(s_partition),
            't_partition': list(t_partition),
            'info': f"Found min cut with value {min_cut_value}"
        })

        return jsonify({
            'algorithm': 'min_cut',
            'category': 'flow',
            'source': source,
            'sink': sink,
            'max_flow': max_flow,  # Min cut value equals max flow
            'min_cut': min_cut_value,
            'cut_edges': [{'from': u, 'to': v} for u, v in cut_edges],
            's_partition': list(s_partition),
            't_partition': list(t_partition),
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@network_flow_routes.route('/api/flow/example_networks')
def get_example_networks():
    """Return example flow networks for testing"""
    examples = [
        {
            "id": "simple_network",
            "name": "Simple Flow Network",
            "graph": {
                "s": {"a": 3, "b": 2},
                "a": {"b": 1, "c": 3},
                "b": {"c": 1, "t": 2},
                "c": {"t": 4},
                "t": {}
            },
            "source": "s",
            "sink": "t"
        },
        {
            "id": "baseball_network",
            "name": "Baseball Elimination Network",
            "graph": {
                "s": {"a": 3, "b": 1, "c": 1},
                "a": {"p1": 2, "p2": 1},
                "b": {"p2": 1},
                "c": {"p3": 1},
                "p1": {"t": 2},
                "p2": {"t": 2},
                "p3": {"t": 1},
                "t": {}
            },
            "source": "s",
            "sink": "t"
        },
        {
            "id": "complex_network",
            "name": "Complex Flow Network",
            "graph": {
                "s": {"a": 10, "c": 5},
                "a": {"b": 9, "d": 15},
                "b": {"c": 4, "e": 15},
                "c": {"a": 3, "d": 4},
                "d": {"e": 8, "f": 10},
                "e": {"t": 10},
                "f": {"b": 6, "t": 10},
                "t": {}
            },
            "source": "s",
            "sink": "t"
        }
    ]

    return jsonify(examples)