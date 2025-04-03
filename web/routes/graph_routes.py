from flask import Blueprint, jsonify, request
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
import heapq

# Import graph algorithms and utils
from kosmos.graph import Graph
from kosmos.graph.traversal import breadth_first_search, depth_first_search
from kosmos.graph import dijkstra
from utils.performance import benchmark, measure_execution_time
from utils.matplotlib_adapter import convert_plot_to_image

# Create the Blueprint
graph_bp = Blueprint('graph', __name__)


@graph_bp.route('/algorithms')
def get_graph_algorithms():
    """Return available graph algorithms"""
    algorithms = [
        {"id": "bfs", "name": "Breadth-First Search", "complexity": "O(V+E)"},
        {"id": "dfs", "name": "Depth-First Search", "complexity": "O(V+E)"},
        {"id": "dijkstra", "name": "Dijkstra's Shortest Path", "complexity": "O(E log V)"},
        {"id": "astar", "name": "A* Search", "complexity": "O(E)"},
        {"id": "prim", "name": "Prim's MST", "complexity": "O(E log V)"},
        {"id": "kruskal", "name": "Kruskal's MST", "complexity": "O(E log E)"}
    ]
    return jsonify(algorithms)


@graph_bp.route('/example-graphs')
def get_example_graphs():
    """Return a list of example graphs"""
    examples = [
        {
            "id": "simple",
            "name": "Simple Graph",
            "vertices": ["A", "B", "C", "D", "E"],
            "edges": [
                {"source": "A", "target": "B", "weight": 1},
                {"source": "A", "target": "C", "weight": 3},
                {"source": "B", "target": "D", "weight": 2},
                {"source": "C", "target": "D", "weight": 1},
                {"source": "D", "target": "E", "weight": 4}
            ],
            "directed": False,
            "weighted": True
        },
        {
            "id": "grid",
            "name": "Grid Graph",
            "vertices": ["1,1", "1,2", "1,3", "2,1", "2,2", "2,3", "3,1", "3,2", "3,3"],
            "edges": [
                {"source": "1,1", "target": "1,2", "weight": 1},
                {"source": "1,2", "target": "1,3", "weight": 1},
                {"source": "2,1", "target": "2,2", "weight": 1},
                {"source": "2,2", "target": "2,3", "weight": 1},
                {"source": "3,1", "target": "3,2", "weight": 1},
                {"source": "3,2", "target": "3,3", "weight": 1},
                {"source": "1,1", "target": "2,1", "weight": 1},
                {"source": "1,2", "target": "2,2", "weight": 1},
                {"source": "1,3", "target": "2,3", "weight": 1},
                {"source": "2,1", "target": "3,1", "weight": 1},
                {"source": "2,2", "target": "3,2", "weight": 1},
                {"source": "2,3", "target": "3,3", "weight": 1}
            ],
            "directed": False,
            "weighted": True
        },
        {
            "id": "complete",
            "name": "Complete Graph",
            "vertices": ["A", "B", "C", "D", "E"],
            "edges": [
                {"source": "A", "target": "B", "weight": 2},
                {"source": "A", "target": "C", "weight": 3},
                {"source": "A", "target": "D", "weight": 1},
                {"source": "A", "target": "E", "weight": 4},
                {"source": "B", "target": "C", "weight": 2},
                {"source": "B", "target": "D", "weight": 5},
                {"source": "B", "target": "E", "weight": 3},
                {"source": "C", "target": "D", "weight": 4},
                {"source": "C", "target": "E", "weight": 1},
                {"source": "D", "target": "E", "weight": 2}
            ],
            "directed": False,
            "weighted": True
        },
        {
            "id": "directed",
            "name": "Directed Graph",
            "vertices": ["A", "B", "C", "D", "E"],
            "edges": [
                {"source": "A", "target": "B", "weight": 1},
                {"source": "A", "target": "C", "weight": 3},
                {"source": "B", "target": "D", "weight": 2},
                {"source": "C", "target": "B", "weight": 1},
                {"source": "D", "target": "E", "weight": 4},
                {"source": "E", "target": "A", "weight": 5}
            ],
            "directed": True,
            "weighted": True
        }
    ]
    return jsonify(examples)


@graph_bp.route('/run', methods=['POST'])
def run_graph_algorithm():
    """Run a graph algorithm and return the results"""
    data = request.json
    algorithm_name = data.get('algorithm', 'bfs')
    graph_data = data.get('graph', {
        'vertices': ['A', 'B', 'C', 'D', 'E'],
        'edges': [
            {'source': 'A', 'target': 'B', 'weight': 1},
            {'source': 'A', 'target': 'C', 'weight': 3},
            {'source': 'B', 'target': 'D', 'weight': 2},
            {'source': 'C', 'target': 'D', 'weight': 1},
            {'source': 'D', 'target': 'E', 'weight': 4}
        ],
        'directed': False,
        'weighted': True
    })

    source = data.get('source', 'A')
    target = data.get('target', 'E')

    # Create a graph from the input data
    graph = create_graph_from_data(graph_data)

    # Map algorithm names to functions
    algorithms = {
        'bfs': breadth_first_search,
        'dfs': depth_first_search,
        'dijkstra': dijkstra
        # Other algorithms are defined locally below
    }

    # Generate visualization frames based on algorithm
    frames = []
    result = None
    execution_time = 0

    try:
        if algorithm_name in algorithms:
            if algorithm_name in ['bfs', 'dfs']:
                # Use the imported algorithm
                result, execution_time = measure_execution_time(algorithms[algorithm_name], graph, source)
                # Generate frames
                frames = generate_traversal_frames(algorithm_name, graph, source, target)
            elif algorithm_name == 'dijkstra':
                # Use the imported algorithm
                result, execution_time = measure_execution_time(algorithms[algorithm_name], graph, source, target)
                # Generate frames
                frames = generate_dijkstra_frames(graph, source, target)
        elif algorithm_name == 'astar':
            # A* is implemented locally
            result, execution_time = measure_execution_time(astar_search, graph, source, target)
            frames = generate_astar_frames(graph, source, target)
        elif algorithm_name == 'prim':
            result, execution_time = measure_execution_time(prim_mst, graph)
            frames = generate_prim_frames(graph)
        elif algorithm_name == 'kruskal':
            result, execution_time = measure_execution_time(kruskal_mst, graph)
            frames = generate_kruskal_frames(graph)
        else:
            return jsonify({'error': 'Algorithm not found'}), 404
    except Exception as e:
        print(f"Error running algorithm: {e}")
        return jsonify({'error': f'Error running algorithm: {str(e)}'}), 500

    # Format the result based on the algorithm
    formatted_result = format_graph_result(algorithm_name, result, source, target)

    return jsonify({
        'algorithm': algorithm_name,
        'result': formatted_result,
        'execution_time': execution_time,
        'visualization': frames,
        'graph': graph_data,
        'category': 'graph'
    })


@graph_bp.route('/compare', methods=['POST'])
def compare_graph_algorithms():
    """Compare multiple graph algorithms"""
    data = request.json
    algorithm_names = data.get('algorithms', ['bfs', 'dfs'])
    graph_data = data.get('graph', {
        'vertices': ['A', 'B', 'C', 'D', 'E'],
        'edges': [
            {'source': 'A', 'target': 'B', 'weight': 1},
            {'source': 'A', 'target': 'C', 'weight': 3},
            {'source': 'B', 'target': 'D', 'weight': 2},
            {'source': 'C', 'target': 'D', 'weight': 1},
            {'source': 'D', 'target': 'E', 'weight': 4}
        ],
        'directed': False,
        'weighted': True
    })

    source = data.get('source', 'A')
    target = data.get('target', 'E')

    # Create a graph from the input data
    graph = create_graph_from_data(graph_data)

    # Map algorithm names to functions
    algorithms = {
        'bfs': breadth_first_search,
        'dfs': depth_first_search,
        'dijkstra': dijkstra,
        'astar': astar_search,
        'prim': prim_mst,
        'kruskal': kruskal_mst
    }

    # Filter out any invalid algorithm names
    selected_algorithms = []
    selected_names = []

    for name in algorithm_names:
        if name in algorithms:
            selected_algorithms.append(algorithms[name])
            selected_names.append(name)

    if not selected_algorithms:
        return jsonify({'error': 'No valid algorithms selected'}), 400

    # Create wrappers for benchmark to ensure consistent function signature
    def create_wrapper(func, algo_name):
        def wrapper(g):
            if algo_name in ['bfs', 'dfs']:
                return func(g, source)
            elif algo_name in ['dijkstra', 'astar']:
                return func(g, source, target)
            else:
                return func(g)

        return wrapper

    wrapped_algorithms = [create_wrapper(algo, name) for algo, name in zip(selected_algorithms, selected_names)]

    # Run benchmark
    results = benchmark(
        wrapped_algorithms,
        [graph],
        labels=selected_names
    )

    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [times[0] for times in results.values()])
    plt.title('Graph Algorithm Performance Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()

    # Convert plot to image
    img_data = convert_plot_to_image(plt)
    plt.close()

    return jsonify({
        'algorithms': selected_names,
        'execution_times': {k: v[0] for k, v in results.items()},
        'comparison_chart': img_data
    })


@graph_bp.route('/code', methods=['GET'])
def get_algorithm_code():
    """Return the code implementation for a graph algorithm"""
    algorithm_name = request.args.get('algorithm', 'bfs')

    code_samples = {
        'bfs': '''
def breadth_first_search(graph, start_vertex):
    """
    Breadth-first search implementation.

    Args:
        graph: The graph to search
        start_vertex: Starting vertex

    Returns:
        Dictionary with visited vertices
    """
    from collections import deque

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
''',
        'dfs': '''
def depth_first_search(graph, start_vertex):
    """
    Depth-first search implementation.

    Args:
        graph: The graph to search
        start_vertex: Starting vertex

    Returns:
        Dictionary with visited vertices
    """
    visited = {}

    def dfs_util(vertex):
        visited[vertex] = True

        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                dfs_util(neighbor)

    dfs_util(start_vertex)
    return visited
''',
        'dijkstra': '''
def dijkstra(graph, start_vertex, end_vertex=None):
    """
    Dijkstra's shortest path algorithm.

    Args:
        graph: The weighted graph
        start_vertex: Starting vertex
        end_vertex: Optional end vertex

    Returns:
        Tuple of (distances, predecessors)
    """
    import heapq

    # Initialize distances and predecessors
    distances = {vertex: float('infinity') for vertex in graph.vertices}
    distances[start_vertex] = 0
    predecessors = {vertex: None for vertex in graph.vertices}

    # Priority queue
    pq = [(0, start_vertex)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        # If we've processed a better path already
        if current_distance > distances[current_vertex]:
            continue

        # Stop if we reached the end vertex
        if end_vertex and current_vertex == end_vertex:
            break

        # Process neighbors
        for neighbor, weight in graph.get_neighbors_with_weights(current_vertex).items():
            distance = current_distance + weight

            # If we found a better path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, predecessors
''',
        'astar': '''
def astar_search(graph, start_vertex, end_vertex):
    """
    A* search algorithm for finding shortest path.

    Args:
        graph: The weighted graph
        start_vertex: Starting vertex
        end_vertex: Target vertex

    Returns:
        Tuple of (path, cost)
    """
    import heapq

    # Heuristic function (estimate of remaining distance)
    def heuristic(vertex):
        # For demonstration, using a simple heuristic (could be improved with coordinates)
        return 0

    # Open set (vertices to explore)
    open_set = [(0 + heuristic(start_vertex), 0, start_vertex, [])]

    # Closed set (vertices already explored)
    closed_set = set()

    while open_set:
        # Get the vertex with lowest f_score (f = g + h)
        f_score, g_score, current, path = heapq.heappop(open_set)

        # If we've reached the goal
        if current == end_vertex:
            path = path + [current]
            return path, g_score

        # If we've already explored this vertex
        if current in closed_set:
            continue

        # Add current to closed set
        closed_set.add(current)

        # Update path
        path = path + [current]

        # Explore neighbors
        for neighbor, weight in graph.get_neighbors_with_weights(current).items():
            if neighbor in closed_set:
                continue

            # Calculate tentative g_score
            tentative_g = g_score + weight

            # Calculate f_score (g + heuristic)
            f = tentative_g + heuristic(neighbor)

            # Add to open set
            heapq.heappush(open_set, (f, tentative_g, neighbor, path))

    # No path found
    return [], float('infinity')
''',
        'prim': '''
def prim_mst(graph):
    """
    Prim's algorithm for Minimum Spanning Tree.

    Args:
        graph: The weighted graph

    Returns:
        Set of edges in the MST
    """
    import heapq

    if not graph.vertices:
        return set()

    # Choose a starting vertex if not provided
    start = next(iter(graph.vertices))

    # Initialize
    mst = []
    visited = {start}

    # Priority queue of edges (weight, u, v)
    edges = []

    # Add all edges from start vertex
    for neighbor, weight in graph.get_neighbors_with_weights(start).items():
        heapq.heappush(edges, (weight, start, neighbor))

    # Grow the MST
    while edges and len(visited) < len(graph.vertices):
        # Get the lowest weight edge
        weight, u, v = heapq.heappop(edges)

        # If we've already visited the destination
        if v in visited:
            continue

        # Add to MST
        mst.append((u, v, weight))
        visited.add(v)

        # Add edges from v to unvisited vertices
        for neighbor, w in graph.get_neighbors_with_weights(v).items():
            if neighbor not in visited:
                heapq.heappush(edges, (w, v, neighbor))

    return mst
''',
        'kruskal': '''
def kruskal_mst(graph):
    """
    Kruskal's algorithm for Minimum Spanning Tree.

    Args:
        graph: The weighted graph

    Returns:
        Set of edges in the MST
    """
    # Union-Find data structure
    parent = {v: v for v in graph.vertices}
    rank = {v: 0 for v in graph.vertices}

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(v1, v2):
        root1 = find(v1)
        root2 = find(v2)

        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1

    # Get all edges and sort by weight
    edges = []
    for (v1, v2), weight in graph.edges.items():
        edges.append((weight, v1, v2))

    edges.sort()

    # MST edges
    mst_edges = set()

    # Process edges in order of weight
    for weight, v1, v2 in edges:
        if find(v1) != find(v2):
            union(v1, v2)
            mst_edges.add((v1, v2, weight))

            # Stop when we have |V|-1 edges
            if len(mst_edges) == len(graph.vertices) - 1:
                break

    return mst_edges
'''
    }

    return jsonify({
        'code': code_samples.get(algorithm_name, 'Code not available for this algorithm')
    })


def create_graph_from_data(graph_data):
    """Create a Graph object from the provided data"""
    try:
        graph = Graph(directed=graph_data.get('directed', False),
                      weighted=graph_data.get('weighted', False))

        # Add vertices
        for vertex in graph_data.get('vertices', []):
            if vertex is not None:  # Skip None vertices
                graph.add_vertex(vertex)

        # Add edges
        for edge in graph_data.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            weight = edge.get('weight', 1)

            # Validate source and target
            if source is not None and target is not None and source in graph.vertices and target in graph.vertices:
                if graph_data.get('weighted', False):
                    graph.add_edge(source, target, weight)
                else:
                    graph.add_edge(source, target)

        return graph
    except Exception as e:
        # If anything goes wrong, return a minimal valid graph
        print(f"Error creating graph: {e}")
        return Graph()


def format_graph_result(algorithm_name, result, source, target):
    """Format the result of a graph algorithm for the API response"""
    if algorithm_name == 'bfs' or algorithm_name == 'dfs':
        return {
            'visited': list(result.keys()) if result else [],
            'starting_vertex': source
        }
    elif algorithm_name == 'dijkstra' or algorithm_name == 'astar':
        if algorithm_name == 'dijkstra':
            if not result:
                return {'distances': {}, 'path': [], 'path_length': None}

            distances, predecessors = result

            # Reconstruct path if target was provided
            path = []
            if target in predecessors:
                current = target
                while current:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()

            return {
                'distances': {k: v if v != float('infinity') else "infinity" for k, v in distances.items()},
                'path': path,
                'path_length': distances[target] if target in distances else None
            }
        else:
            if not result:
                return {'path': [], 'path_length': None}

            path, cost = result
            return {
                'path': path,
                'path_length': cost
            }
    elif algorithm_name == 'prim' or algorithm_name == 'kruskal':
        if not result:
            return {'mst_edges': [], 'total_weight': 0}

        mst_edges = result
        return {
            'mst_edges': [{'source': v1, 'target': v2, 'weight': w} for v1, v2, w in mst_edges],
            'total_weight': sum(w for _, _, w in mst_edges)
        }
    else:
        return result


def generate_traversal_frames(algorithm_name, graph, start_vertex, end_vertex):
    """Generate visualization frames for BFS or DFS traversal"""
    frames = []

    # Validate inputs
    if not graph or not hasattr(graph, 'vertices') or not graph.vertices:
        return [{'info': 'Empty graph', 'visited': [], 'current': None, 'queue': []}]

    if start_vertex is None or start_vertex not in graph.vertices:
        # Use first vertex if none specified or invalid
        start_vertex = next(iter(graph.vertices)) if graph.vertices else None

    if start_vertex is None:
        return [{'info': 'No valid start vertex found', 'visited': [], 'current': None, 'queue': []}]

    # Common setup
    frames.append({
        'info': f'Starting {algorithm_name.upper()} from vertex {start_vertex}',
        'visited': [],
        'current': None,
        'queue': [start_vertex] if algorithm_name == 'bfs' else [],
        'start': start_vertex,
        'end': end_vertex
    })

    if algorithm_name == 'bfs':
        # BFS implementation with frame generation
        visited = {}
        queue = deque([start_vertex])
        visited[start_vertex] = True

        frames.append({
            'info': f'Added {start_vertex} to the queue',
            'visited': list(visited.keys()),
            'current': None,
            'queue': list(queue),
            'start': start_vertex,
            'end': end_vertex
        })

        while queue:
            vertex = queue.popleft()

            # Skip invalid vertices
            if vertex not in graph.vertices:
                continue

            frames.append({
                'info': f'Dequeued {vertex}',
                'visited': list(visited.keys()),
                'current': vertex,
                'queue': list(queue),
                'start': start_vertex,
                'end': end_vertex
            })

            # Handle case where vertex is not in adjacency list
            neighbors = graph.get_neighbors(vertex) if hasattr(graph, 'get_neighbors') else []

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = True
                    queue.append(neighbor)

                    # Record edge being considered
                    frames.append({
                        'info': f'Discovered {neighbor} from {vertex}',
                        'visited': list(visited.keys()),
                        'current': vertex,
                        'queue': list(queue),
                        'consideredEdges': [f'{vertex},{neighbor}'],
                        'start': start_vertex,
                        'end': end_vertex
                    })

            frames.append({
                'info': f'Finished processing {vertex}',
                'visited': list(visited.keys()),
                'current': None,
                'queue': list(queue),
                'start': start_vertex,
                'end': end_vertex
            })

        # Construct path if target is specified
        if end_vertex is not None and end_vertex in graph.vertices:
            # For simplicity, we'll just add a frame showing if path exists
            if end_vertex in visited:
                frames.append({
                    'info': f'Path exists from {start_vertex} to {end_vertex}',
                    'visited': list(visited.keys()),
                    'current': None,
                    'queue': [],
                    'start': start_vertex,
                    'end': end_vertex
                })
            else:
                frames.append({
                    'info': f'No path found from {start_vertex} to {end_vertex}',
                    'visited': list(visited.keys()),
                    'current': None,
                    'queue': [],
                    'start': start_vertex,
                    'end': end_vertex
                })

        frames.append({
            'info': 'BFS complete',
            'visited': list(visited.keys()),
            'current': None,
            'queue': [],
            'start': start_vertex,
            'end': end_vertex
        })
    else:
        # DFS implementation with frame generation
        visited = {}
        stack = [start_vertex]  # Using a stack for iterative DFS (easier to visualize)

        frames.append({
            'info': f'Added {start_vertex} to the stack',
            'visited': [],
            'current': None,
            'queue': list(stack),  # Reusing queue for visualization
            'start': start_vertex,
            'end': end_vertex
        })

        while stack:
            vertex = stack.pop()

            # Skip invalid vertices
            if vertex not in graph.vertices:
                continue

            if vertex not in visited:
                visited[vertex] = True

                frames.append({
                    'info': f'Popped {vertex} from stack',
                    'visited': list(visited.keys()),
                    'current': vertex,
                    'queue': list(stack),
                    'start': start_vertex,
                    'end': end_vertex
                })

                # Get neighbors in reverse order for stack (to match recursive DFS)
                neighbors = list(graph.get_neighbors(vertex)) if hasattr(graph, 'get_neighbors') else []
                neighbors.reverse()

                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)

                        frames.append({
                            'info': f'Added {neighbor} to stack',
                            'visited': list(visited.keys()),
                            'current': vertex,
                            'queue': list(stack),
                            'consideredEdges': [f'{vertex},{neighbor}'],
                            'start': start_vertex,
                            'end': end_vertex
                        })

            frames.append({
                'info': f'Finished processing {vertex}',
                'visited': list(visited.keys()),
                'current': None,
                'queue': list(stack),
                'start': start_vertex,
                'end': end_vertex
            })

        # Similar path check as in BFS
        if end_vertex is not None and end_vertex in graph.vertices:
            if end_vertex in visited:
                frames.append({
                    'info': f'Path exists from {start_vertex} to {end_vertex}',
                    'visited': list(visited.keys()),
                    'current': None,
                    'queue': [],
                    'start': start_vertex,
                    'end': end_vertex
                })
            else:
                frames.append({
                    'info': f'No path found from {start_vertex} to {end_vertex}',
                    'visited': list(visited.keys()),
                    'current': None,
                    'queue': [],
                    'start': start_vertex,
                    'end': end_vertex
                })

        frames.append({
            'info': 'DFS complete',
            'visited': list(visited.keys()),
            'current': None,
            'queue': [],
            'start': start_vertex,
            'end': end_vertex
        })

    return frames


def generate_dijkstra_frames(graph, start_vertex, end_vertex):
    """Generate visualization frames for Dijkstra's algorithm"""
    frames = []

    # Validate inputs
    if not graph or not hasattr(graph, 'vertices') or not graph.vertices:
        return [{'info': 'Empty graph', 'visited': [], 'current': None, 'queue': []}]

    if start_vertex is None or start_vertex not in graph.vertices:
        start_vertex = next(iter(graph.vertices)) if graph.vertices else None

    if start_vertex is None:
        return [{'info': 'No valid start vertex found', 'visited': [], 'current': None, 'queue': []}]

    # Initialize
    distances = {vertex: float('infinity') for vertex in graph.vertices}
    distances[start_vertex] = 0
    predecessors = {vertex: None for vertex in graph.vertices}

    # Priority queue (distance, vertex)
    pq = [(0, start_vertex)]
    heapq.heapify(pq)

    # Track visited nodes
    visited = set()

    # Initial frame
    frames.append({
        'info': f'Starting Dijkstra\'s algorithm from {start_vertex}',
        'visited': [],
        'current': None,
        'queue': [start_vertex],
        'distances': {k: 'infinity' if v == float('infinity') else v for k, v in distances.items()},
        'start': start_vertex,
        'end': end_vertex
    })

    while pq:
        dist, vertex = heapq.heappop(pq)

        # Skip if we've found a better path already
        if dist > distances.get(vertex, float('infinity')):
            continue

        # Skip invalid vertices
        if vertex not in graph.vertices:
            continue

        # Mark as visited
        visited.add(vertex)

        frames.append({
            'info': f'Processing vertex {vertex} with distance {dist}',
            'visited': list(visited),
            'current': vertex,
            'queue': [v for _, v in pq],
            'distances': {k: 'infinity' if v == float('infinity') else v for k, v in distances.items()},
            'start': start_vertex,
            'end': end_vertex
        })

        # Stop if we reached the end vertex
        if vertex == end_vertex:
            frames.append({
                'info': f'Reached target {end_vertex} with distance {distances[end_vertex]}',
                'visited': list(visited),
                'current': vertex,
                'queue': [v for _, v in pq],
                'distances': {k: 'infinity' if v == float('infinity') else v for k, v in distances.items()},
                'start': start_vertex,
                'end': end_vertex
            })
            break

        # Process neighbors
        try:
            # Handle potential errors in get_neighbors_with_weights
            if hasattr(graph, 'get_neighbors_with_weights'):
                neighbors_with_weights = graph.get_neighbors_with_weights(vertex)
            else:
                neighbors_with_weights = {}
                for neighbor in graph.get_neighbors(vertex):
                    weight = graph.get_edge_weight(vertex, neighbor) if hasattr(graph, 'get_edge_weight') else 1
                    neighbors_with_weights[neighbor] = weight

            for neighbor, weight in neighbors_with_weights.items():
                if neighbor in visited:
                    continue

                new_dist = distances[vertex] + weight

                # Add this edge to consideration
                frames.append({
                    'info': f'Considering edge from {vertex} to {neighbor} with weight {weight}',
                    'visited': list(visited),
                    'current': vertex,
                    'queue': [v for _, v in pq],
                    'distances': {k: 'infinity' if v == float('infinity') else v for k, v in distances.items()},
                    'consideredEdges': [f'{vertex},{neighbor}'],
                    'start': start_vertex,
                    'end': end_vertex
                })

                if new_dist < distances.get(neighbor, float('infinity')):
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = vertex
                    heapq.heappush(pq, (new_dist, neighbor))

                    frames.append({
                        'info': f'Found better path to {neighbor} with distance {new_dist}',
                        'visited': list(visited),
                        'current': vertex,
                        'queue': [v for _, v in pq],
                        'distances': {k: 'infinity' if v == float('infinity') else v for k, v in distances.items()},
                        'start': start_vertex,
                        'end': end_vertex
                    })
        except Exception as e:
            # If there's an error processing neighbors, add an error frame
            frames.append({
                'info': f'Error processing neighbors of {vertex}: {str(e)}',
                'visited': list(visited),
                'current': vertex,
                'queue': [v for _, v in pq],
                'distances': {k: 'infinity' if v == float('infinity') else v for k, v in distances.items()},
                'start': start_vertex,
                'end': end_vertex
            })

    # Reconstruct path if target was provided
    if end_vertex in predecessors and predecessors[end_vertex] is not None:
        path = []
        current = end_vertex
        while current:
            path.append(current)
            current = predecessors.get(current)
        path.reverse()

        # Path edges
        path_edges = []
        for i in range(len(path) - 1):
            path_edges.append(f'{path[i]},{path[i + 1]}')

        frames.append({
            'info': f'Shortest path found: {" -> ".join(path)} with distance {distances[end_vertex]}',
            'visited': list(visited),
            'current': None,
            'queue': [],
            'distances': {k: 'infinity' if v == float('infinity') else v for k, v in distances.items()},
            'path': path,
            'pathEdges': path_edges,
            'start': start_vertex,
            'end': end_vertex
        })
    elif end_vertex in graph.vertices:
        frames.append({
            'info': f'No path found to {end_vertex}',
            'visited': list(visited),
            'current': None,
            'queue': [],
            'distances': {k: 'infinity' if v == float('infinity') else v for k, v in distances.items()},
            'start': start_vertex,
            'end': end_vertex
        })

    return frames


def generate_astar_frames(graph, start_vertex, end_vertex):
    """Generate visualization frames for A* search algorithm"""
    frames = []

    # Validate inputs
    if not graph or not hasattr(graph, 'vertices') or not graph.vertices:
        return [{'info': 'Empty graph', 'visited': [], 'current': None, 'queue': []}]

    if start_vertex is None or start_vertex not in graph.vertices:
        start_vertex = next(iter(graph.vertices)) if graph.vertices else None

    if start_vertex is None:
        return [{'info': 'No valid start vertex found', 'visited': [], 'current': None, 'queue': []}]

    if end_vertex is None or end_vertex not in graph.vertices:
        return [{'info': 'No valid target vertex found', 'visited': [], 'current': None, 'queue': []}]

    # Simple heuristic function (in reality, you'd use coordinates)
    def heuristic(vertex):
        return 0

    # Open set: (f_score, g_score, vertex, path)
    open_set = [(heuristic(start_vertex), 0, start_vertex, [])]
    heapq.heapify(open_set)

    # Closed set
    closed_set = set()

    # For visualization
    g_scores = {vertex: float('infinity') for vertex in graph.vertices}
    g_scores[start_vertex] = 0

    # Initial frame
    frames.append({
        'info': f'Starting A* search from {start_vertex} to {end_vertex}',
        'visited': [],
        'current': None,
        'queue': [start_vertex],
        'distances': {k: 'infinity' if v == float('infinity') else v for k, v in g_scores.items()},
        'start': start_vertex,
        'end': end_vertex
    })

    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)

        # If we've already visited this vertex, skip
        if current in closed_set:
            continue

        frames.append({
            'info': f'Exploring vertex {current} with f_score={f_score} (g={g_score}, h={f_score - g_score})',
            'visited': list(closed_set),
            'current': current,
            'queue': [item[2] for item in open_set],
            'distances': {k: 'infinity' if v == float('infinity') else v for k, v in g_scores.items()},
            'start': start_vertex,
            'end': end_vertex
        })

        # If we've reached the goal
        if current == end_vertex:
            final_path = path + [current]

            # Path edges
            path_edges = []
            for i in range(len(final_path) - 1):
                path_edges.append(f'{final_path[i]},{final_path[i + 1]}')

            frames.append({
                'info': f'Found path: {" -> ".join(final_path)} with cost {g_score}',
                'visited': list(closed_set),
                'current': current,
                'queue': [item[2] for item in open_set],
                'distances': {k: 'infinity' if v == float('infinity') else v for k, v in g_scores.items()},
                'path': final_path,
                'pathEdges': path_edges,
                'start': start_vertex,
                'end': end_vertex
            })
            break

        # Mark as visited
        closed_set.add(current)

        # Updated path
        new_path = path + [current]

        # Explore neighbors
        try:
            # Handle potential errors in get_neighbors_with_weights
            if hasattr(graph, 'get_neighbors_with_weights'):
                neighbors_with_weights = graph.get_neighbors_with_weights(current)
            else:
                neighbors_with_weights = {}
                for neighbor in graph.get_neighbors(current):
                    weight = graph.get_edge_weight(current, neighbor) if hasattr(graph, 'get_edge_weight') else 1
                    neighbors_with_weights[neighbor] = weight

            for neighbor, weight in neighbors_with_weights.items():
                if neighbor in closed_set:
                    continue

                # Add edge being considered
                frames.append({
                    'info': f'Considering edge from {current} to {neighbor} with weight {weight}',
                    'visited': list(closed_set),
                    'current': current,
                    'queue': [item[2] for item in open_set],
                    'distances': {k: 'infinity' if v == float('infinity') else v for k, v in g_scores.items()},
                    'consideredEdges': [f'{current},{neighbor}'],
                    'start': start_vertex,
                    'end': end_vertex
                })

                # Calculate new scores
                tentative_g = g_score + weight

                if tentative_g < g_scores.get(neighbor, float('infinity')):
                    g_scores[neighbor] = tentative_g

                    # Calculate f_score (g + heuristic)
                    f = tentative_g + heuristic(neighbor)

                    # Add to open set
                    heapq.heappush(open_set, (f, tentative_g, neighbor, new_path))

                    frames.append({
                        'info': f'Added {neighbor} to open set with f_score={f} (g={tentative_g}, h={f - tentative_g})',
                        'visited': list(closed_set),
                        'current': current,
                        'queue': [item[2] for item in open_set],
                        'distances': {k: 'infinity' if v == float('infinity') else v for k, v in g_scores.items()},
                        'start': start_vertex,
                        'end': end_vertex
                    })
        except Exception as e:
            # If there's an error processing neighbors, add an error frame
            frames.append({
                'info': f'Error processing neighbors of {current}: {str(e)}',
                'visited': list(closed_set),
                'current': current,
                'queue': [item[2] for item in open_set],
                'distances': {k: 'infinity' if v == float('infinity') else v for k, v in g_scores.items()},
                'start': start_vertex,
                'end': end_vertex
            })

    # No path found
    if len(frames) > 0 and 'path' not in frames[-1]:
        frames.append({
            'info': f'No path found from {start_vertex} to {end_vertex}',
            'visited': list(closed_set),
            'current': None,
            'queue': [],
            'distances': {k: 'infinity' if v == float('infinity') else v for k, v in g_scores.items()},
            'start': start_vertex,
            'end': end_vertex
        })

    return frames


# Implementations of additional algorithms

def astar_search(graph, start_vertex, end_vertex):
    """A* search algorithm for finding shortest path"""

    # For this implementation, we'll use a simple heuristic
    # In a real implementation, you'd want a better heuristic based on coordinates
    def heuristic(vertex):
        return 0

    # Priority queue for the open set
    # Format: (f_score, g_score, vertex, path)
    open_set = [(heuristic(start_vertex), 0, start_vertex, [])]
    heapq.heapify(open_set)

    # Set of visited vertices
    closed_set = set()

    while open_set:
        # Get the vertex with lowest f_score
        f_score, g_score, current, path = heapq.heappop(open_set)

        # If we've reached the goal, return the path
        if current == end_vertex:
            return path + [current], g_score

        # Skip if we've already visited this vertex
        if current in closed_set:
            continue

        # Mark as visited
        closed_set.add(current)

        # New path up to current
        new_path = path + [current]

        # Explore neighbors
        for neighbor, weight in graph.get_neighbors_with_weights(current).items():
            if neighbor in closed_set:
                continue

            # Calculate g and f scores
            new_g_score = g_score + weight
            new_f_score = new_g_score + heuristic(neighbor)

            # Add to open set
            heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, new_path))

    # No path found
    return [], float('infinity')


def prim_mst(graph):
    """Prim's algorithm for Minimum Spanning Tree"""
    if not graph.vertices:
        return set()

    # Start with an arbitrary vertex
    start = next(iter(graph.vertices))

    # Track visited vertices
    visited = {start}

    # Edges in the MST
    mst_edges = set()

    # Priority queue for edges: (weight, v1, v2)
    edges = []

    # Add edges from start vertex
    for neighbor, weight in graph.get_neighbors_with_weights(start).items():
        heapq.heappush(edges, (weight, start, neighbor))

    # Build the MST
    while edges and len(visited) < len(graph.vertices):
        weight, v1, v2 = heapq.heappop(edges)

        if v2 in visited:
            continue

        mst_edges.add((v1, v2, weight))
        visited.add(v2)

        # Add edges from newly added vertex
        for neighbor, w in graph.get_neighbors_with_weights(v2).items():
            if neighbor not in visited:
                heapq.heappush(edges, (w, v2, neighbor))

    return mst_edges


def kruskal_mst(graph):
    """Kruskal's algorithm for Minimum Spanning Tree"""
    # Union-Find data structure
    parent = {v: v for v in graph.vertices}
    rank = {v: 0 for v in graph.vertices}

    # Find operation with path compression
    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    # Union operation with rank
    def union(v1, v2):
        root1 = find(v1)
        root2 = find(v2)

        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1

    # Get all edges and sort by weight
    edges = []
    for u, v in graph.edges:
        weight = graph.get_edge_weight(u, v)
        edges.append((weight, u, v))

    # Sort edges by weight
    edges.sort()

    # MST edges
    mst_edges = set()

    # Process edges in order of weight
    for weight, v1, v2 in edges:
        if find(v1) != find(v2):
            union(v1, v2)
            mst_edges.add((v1, v2, weight))

            # Stop when we have |V|-1 edges
            if len(mst_edges) == len(graph.vertices) - 1:
                break

    return mst_edges


def generate_prim_frames(graph):
    """Generate visualization frames for Prim's MST algorithm"""
    frames = []

    # Validate inputs
    if not graph or not hasattr(graph, 'vertices') or not graph.vertices:
        return [{'info': 'Empty graph', 'visited': [], 'current': None, 'pathEdges': []}]

    # Start with an arbitrary vertex
    start = next(iter(graph.vertices))

    # Visited vertices
    visited = {start}

    # MST edges
    mst_edges = set()

    # Priority queue: (weight, v1, v2)
    edges = []

    # Initial frame
    frames.append({
        'info': f'Starting Prim\'s algorithm from vertex {start}',
        'visited': list(visited),
        'current': start,
        'pathEdges': []
    })

    # Add edges from start
    try:
        for neighbor, weight in graph.get_neighbors_with_weights(start).items():
            heapq.heappush(edges, (weight, start, neighbor))

            frames.append({
                'info': f'Added edge ({start},{neighbor}) with weight {weight} to the priority queue',
                'visited': list(visited),
                'current': start,
                'queue': [f'{v1}-{v2}' for _, v1, v2 in edges],
                'consideredEdges': [f'{start},{neighbor}'],
                'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
            })
    except Exception as e:
        frames.append({
            'info': f'Error adding edges from start vertex: {str(e)}',
            'visited': list(visited),
            'current': start,
            'pathEdges': []
        })
        return frames

    # Build MST
    while edges and len(visited) < len(graph.vertices):
        weight, v1, v2 = heapq.heappop(edges)

        frames.append({
            'info': f'Considering minimum edge ({v1},{v2}) with weight {weight}',
            'visited': list(visited),
            'current': v1,
            'queue': [f'{v1}-{v2}' for _, v1, v2 in edges],
            'consideredEdges': [f'{v1},{v2}'],
            'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
        })

        if v2 in visited:
            frames.append({
                'info': f'Skipping edge ({v1},{v2}) as {v2} is already in the MST',
                'visited': list(visited),
                'current': v1,
                'queue': [f'{v1}-{v2}' for _, v1, v2 in edges],
                'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
            })
            continue

        # Add to MST
        mst_edges.add((v1, v2, weight))
        visited.add(v2)

        frames.append({
            'info': f'Added edge ({v1},{v2}) with weight {weight} to the MST',
            'visited': list(visited),
            'current': v2,
            'queue': [f'{v1}-{v2}' for _, v1, v2 in edges],
            'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
        })

        # Add edges from v2
        try:
            for neighbor, w in graph.get_neighbors_with_weights(v2).items():
                if neighbor not in visited:
                    heapq.heappush(edges, (w, v2, neighbor))

                    frames.append({
                        'info': f'Added edge ({v2},{neighbor}) with weight {w} to the priority queue',
                        'visited': list(visited),
                        'current': v2,
                        'queue': [f'{v1}-{v2}' for _, v1, v2 in edges],
                        'consideredEdges': [f'{v2},{neighbor}'],
                        'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
                    })
        except Exception as e:
            frames.append({
                'info': f'Error adding edges from vertex {v2}: {str(e)}',
                'visited': list(visited),
                'current': v2,
                'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
            })

    # Final MST
    total_weight = sum(w for _, _, w in mst_edges)

    frames.append({
        'info': f'MST complete with total weight {total_weight}',
        'visited': list(visited),
        'current': None,
        'queue': [],
        'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
    })

    return frames


def generate_kruskal_frames(graph):
    """Generate visualization frames for Kruskal's MST algorithm"""
    frames = []

    # Validate inputs
    if not graph or not hasattr(graph, 'vertices') or not graph.vertices:
        return [{'info': 'Empty graph', 'visited': [], 'current': None, 'pathEdges': []}]

    # Union-Find data structure
    parent = {v: v for v in graph.vertices}
    rank = {v: 0 for v in graph.vertices}

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(v1, v2):
        root1 = find(v1)
        root2 = find(v2)

        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1

    # Get all edges and sort by weight
    edges = []

    try:
        for u, v in graph.edges:
            weight = graph.get_edge_weight(u, v)
            edges.append((weight, u, v))

        # Sort edges
        edges.sort()
    except Exception as e:
        return [{
            'info': f'Error getting edges: {str(e)}',
            'visited': [],
            'current': None,
            'pathEdges': []
        }]

    # Initial frame
    frames.append({
        'info': 'Starting Kruskal\'s algorithm',
        'visited': [],
        'current': None,
        'pathEdges': []
    })

    # Show sorted edges
    edge_strs = [f'({v1},{v2}): {w}' for w, v1, v2 in edges]
    frames.append({
        'info': f'Sorted edges by weight: {", ".join(edge_strs)}',
        'visited': [],
        'current': None,
        'pathEdges': []
    })

    # MST edges
    mst_edges = set()

    # Process edges
    for weight, v1, v2 in edges:
        frames.append({
            'info': f'Considering edge ({v1},{v2}) with weight {weight}',
            'visited': [v for v in graph.vertices if find(v) != v],  # Show merged components
            'current': None,
            'consideredEdges': [f'{v1},{v2}'],
            'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
        })

        if find(v1) != find(v2):
            union(v1, v2)
            mst_edges.add((v1, v2, weight))

            frames.append({
                'info': f'Added edge ({v1},{v2}) with weight {weight} to the MST',
                'visited': [v for v in graph.vertices if find(v) != v],
                'current': None,
                'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
            })

            # Stop when we have |V|-1 edges
            if len(mst_edges) == len(graph.vertices) - 1:
                break
        else:
            frames.append({
                'info': f'Skipping edge ({v1},{v2}) as it would create a cycle',
                'visited': [v for v in graph.vertices if find(v) != v],
                'current': None,
                'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
            })

    # Final MST
    total_weight = sum(w for _, _, w in mst_edges)

    frames.append({
        'info': f'MST complete with total weight {total_weight}',
        'visited': [v for v in graph.vertices if find(v) != v],
        'current': None,
        'queue': [],
        'pathEdges': [f'{v1},{v2}' for v1, v2, _ in mst_edges]
    })

    return frames