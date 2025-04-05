from flask import Blueprint, jsonify, request
from collections import defaultdict

from kosmos.network_flow.base import FlowNetwork, FordFulkerson, EdmondsKarp
from kosmos.network_flow.min_cut import MinCut

network_flow_routes = Blueprint('network_flow_routes', __name__)


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
                'residual_graph': {u: dict(flow_network.graph[u]) for u in flow_network.vertices},
                'flow_graph': {u: dict(flow_graph[u]) for u in flow_graph}
            })

        # Convert flow graph to a cleaner format
        final_flow = {}
        for u in flow_network.vertices:
            final_flow[u] = {}
            for v in flow_network.graph[u]:
                if flow_graph[u][v] > 0:
                    final_flow[u][v] = flow_graph[u][v]

        return jsonify({
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
                'flow_graph': {u: dict(flow_graph[u]) for u in flow_graph}
            })

        # Convert flow graph to a cleaner format
        final_flow = {}
        for u in flow_network.vertices:
            final_flow[u] = {}
            for v in flow_network.graph[u]:
                if flow_graph[u][v] > 0:
                    final_flow[u][v] = flow_graph[u][v]

        return jsonify({
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

        return jsonify({
            'max_flow': min_cut_value,  # Min cut value equals max flow
            'min_cut': min_cut_value,
            'cut_edges': [{'from': u, 'to': v} for u, v in cut_edges],
            's_partition': list(s_partition),
            't_partition': list(t_partition)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500