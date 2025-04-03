from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import networkx as nx
from typing import List, Any, Callable, Tuple, Optional
import math

def visualize_graph(graph, layout='spring', node_labels=True, edge_labels=True,
                    node_color='lightblue', edge_color='black', figsize=(10, 8)):
    """
    Visualize a graph using matplotlib.

    Args:
        graph: The graph to visualize (must have vertices, edges, get_edge_weight methods)
        layout: The layout algorithm ('spring', 'circular', 'random', etc.)
        node_labels: Whether to show node labels
        edge_labels: Whether to show edge labels
        node_color: Color for the nodes
        edge_color: Color for the edges
        figsize: Figure size as (width, height) tuple
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required for graph visualization. Install with: pip install networkx")
        return

    # Convert to NetworkX graph
    if hasattr(graph, 'directed') and graph.directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add nodes
    for vertex in graph.vertices:
        G.add_node(vertex)

    # Add edges with weights if available
    if hasattr(graph, 'weighted') and graph.weighted:
        for (u, v), weight in graph.edges.items():
            G.add_edge(u, v, weight=weight)
    else:
        for (u, v) in graph.edges:
            G.add_edge(u, v)

    # Create layout
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Draw graph
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=1.5,
                           arrowstyle='-|>' if hasattr(graph, 'directed') and graph.directed else '-',
                           arrowsize=15)

    # Add labels if requested
    if node_labels:
        nx.draw_networkx_labels(G, pos, font_size=12)

    if edge_labels and hasattr(graph, 'weighted') and graph.weighted:
        edge_labels_dict = {(u, v): f"{w}" for (u, v), w in graph.edges.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_size=10)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_tree(root, node_radius=0.5, level_spacing=2.0, figsize=(10, 8)):
    """
    Visualize a binary tree using matplotlib.

    Args:
        root: The root node of the tree (must have left and right attributes)
        node_radius: Radius of the circles representing nodes
        level_spacing: Vertical spacing between tree levels
        figsize: Figure size as (width, height) tuple
    """
    if root is None:
        print("Empty tree")
        return

    # Calculate tree dimensions
    def height(node):
        if node is None:
            return 0
        return 1 + max(height(node.left), height(node.right))

    def width(node):
        if node is None:
            return 0
        return 1 + width(node.left) + width(node.right)

    tree_height = height(root)
    tree_width = width(root)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-tree_width, tree_width)
    ax.set_ylim(-tree_height * level_spacing, level_spacing)
    ax.axis('off')

    # Draw the tree recursively
    def draw_node(node, x, y, dx):
        if node is None:
            return

        # Draw the node
        circle = plt.Circle((x, y), node_radius, fill=True, color='lightblue', ec='black')
        ax.add_patch(circle)

        # Add the node value
        ax.text(x, y, str(node.value), ha='center', va='center')

        # Calculate new_dx (half the horizontal distance to the next node)
        new_dx = dx / 2

        # Draw left child and edge if it exists
        if node.left:
            left_x = x - dx
            left_y = y - level_spacing
            ax.plot([x, left_x], [y - node_radius, left_y + node_radius], 'k-')
            draw_node(node.left, left_x, left_y, new_dx)

        # Draw right child and edge if it exists
        if node.right:
            right_x = x + dx
            right_y = y - level_spacing
            ax.plot([x, right_x], [y - node_radius, right_y + node_radius], 'k-')
            draw_node(node.right, right_x, right_y, new_dx)

    # Calculate initial dx (half the width of the tree)
    initial_dx = 2 ** (tree_height - 1)

    # Draw the tree starting from the root
    draw_node(root, 0, 0, initial_dx)

    plt.tight_layout()
    plt.show()


def visualize_algorithm_steps(algorithm: Callable, input_data: Any, steps: List[Any],
                              title: str = "Algorithm Visualization", figsize=(10, 6)):
    """
    Visualize steps of an algorithm's execution.

    Args:
        algorithm: The algorithm function to visualize
        input_data: The input data for the algorithm
        steps: List of intermediate steps to display
        title: Title for the visualization
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)
    plt.suptitle(title)

    # Create subplots for each step
    n_steps = len(steps)
    rows = (n_steps + 2) // 3  # +2 to include input and output
    cols = min(3, n_steps + 2)

    # Plot input
    plt.subplot(rows, cols, 1)
    plt.title("Input")
    _plot_data(input_data)

    # Plot steps
    for i, step in enumerate(steps):
        plt.subplot(rows, cols, i + 2)
        plt.title(f"Step {i + 1}")
        _plot_data(step)

    # Plot output
    plt.subplot(rows, cols, n_steps + 2)
    plt.title("Output")
    result = algorithm(input_data)
    _plot_data(result)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return result


def _plot_data(data):
    """Helper function to plot different types of data."""
    if isinstance(data, (list, np.ndarray)) and all(isinstance(x, (int, float)) for x in data):
        # Numeric array
        plt.bar(range(len(data)), data, color='skyblue')
        plt.xticks(range(len(data)), [str(i) for i in range(len(data))])
    elif isinstance(data, dict):
        # Dictionary
        plt.bar(range(len(data)), list(data.values()), color='skyblue')
        plt.xticks(range(len(data)), list(data.keys()), rotation=45)
    elif hasattr(data, 'vertices') and hasattr(data, 'edges'):
        # Graph-like object
        visualize_graph(data, figsize=(4, 4))
    else:
        # Default: just show the text representation
        plt.text(0.5, 0.5, str(data), ha='center', va='center')
        plt.axis('off')


def visualize_sorting_algorithm(algorithm: Callable, arr: List[int], title: str = None,
                                show_comparisons=True, interval=200):
    """
    Visualize a sorting algorithm using an animation.

    Args:
        algorithm: The sorting algorithm function
        arr: The array to sort
        title: The title of the animation
        show_comparisons: Whether to highlight elements being compared
        interval: Time interval between frames in milliseconds

    Returns:
        The animation object
    """
    # Copy the array to avoid modifying the original
    arr = arr.copy()

    # Get all steps in the sorting process
    history = []
    comparisons = []
    swaps = []

    # Monkey patch comparison and swap operations
    def compare(i, j, array):
        comparisons.append((i, j))
        return array[i] > array[j]

    def swap(i, j, array):
        array[i], array[j] = array[j], array[i]
        swaps.append((i, j))
        history.append(array.copy())

    # Wrap the algorithm to record steps
    def sorting_algorithm_with_history(arr):
        # Save initial state
        history.append(arr.copy())

        # Define algorithm-specific instrumentation
        if algorithm.__name__ == 'bubble_sort':
            def instrumented_bubble_sort(arr):
                n = len(arr)
                arr_copy = arr.copy()

                for i in range(n):
                    for j in range(0, n - i - 1):
                        comparisons.append((j, j + 1))
                        if arr_copy[j] > arr_copy[j + 1]:
                            arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                            swaps.append((j, j + 1))
                            history.append(arr_copy.copy())

                return arr_copy

            return instrumented_bubble_sort(arr)
        elif algorithm.__name__ == 'insertion_sort':
            def instrumented_insertion_sort(arr):
                arr_copy = arr.copy()

                for i in range(1, len(arr_copy)):
                    key = arr_copy[i]
                    j = i - 1
                    comparisons.append((j, i))

                    while j >= 0 and arr_copy[j] > key:
                        arr_copy[j + 1] = arr_copy[j]
                        swaps.append((j, j + 1))
                        history.append(arr_copy.copy())
                        j -= 1
                        if j >= 0:
                            comparisons.append((j, i))

                    arr_copy[j + 1] = key
                    if j + 1 != i:
                        history.append(arr_copy.copy())

                return arr_copy

            return instrumented_insertion_sort(arr)
        else:
            # Default instrumentation - just run the algorithm
            result = algorithm(arr)
            if result is not None:
                history.append(result.copy())
            else:
                # If the algorithm modifies in-place and returns None
                history.append(arr.copy())
            return result

    sorting_algorithm_with_history(arr)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    if title:
        ax.set_title(title)

    # Initial state
    bars = ax.bar(range(len(arr)), history[0], color='skyblue', edgecolor='black')
    ax.set_xlim(-0.5, len(arr) - 0.5)
    ax.set_ylim(0, max(history[0]) * 1.1 if history[0] else 10)
    ax.set_xticks(range(len(arr)))

    # Add info text
    info_text = ax.text(len(arr) / 2, max(history[0]) * 1.05, 'Initial state',
                        ha='center', va='center', fontsize=12)

    # Create colors for different states
    colors = {
        'default': 'skyblue',
        'compare': 'yellow',
        'swap': 'red',
        'sorted': 'green'
    }

    # Track the current frame for comparison info
    current_frame = [0]

    # Update function for animation
    def update(frame):
        current_frame[0] = frame
        frame_info = ""

        # Update bar heights
        for i, val in enumerate(history[min(frame, len(history) - 1)]):
            bars[i].set_height(val)
            bars[i].set_color(colors['default'])

        # Color for comparisons
        if show_comparisons and frame > 0 and frame - 1 < len(comparisons):
            i, j = comparisons[frame - 1]
            if 0 <= i < len(arr) and 0 <= j < len(arr):
                bars[i].set_color(colors['compare'])
                bars[j].set_color(colors['compare'])
                frame_info = f"Comparing {history[min(frame, len(history) - 1)][i]} and {history[min(frame, len(history) - 1)][j]}"

        # Color for swaps
        if frame > 0 and frame - 1 < len(swaps):
            i, j = swaps[frame - 1]
            if 0 <= i < len(arr) and 0 <= j < len(arr):
                bars[i].set_color(colors['swap'])
                bars[j].set_color(colors['swap'])
                frame_info = f"Swapping {history[min(frame, len(history) - 1)][i]} and {history[min(frame, len(history) - 1)][j]}"

        # Final sorted state
        if frame >= len(history) - 1:
            for i in range(len(arr)):
                bars[i].set_color(colors['sorted'])
            frame_info = "Sorted!"

        # Update info text
        info_text.set_text(frame_info)

        return bars + [info_text]

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(history) + len(comparisons),
                         interval=interval, repeat=False, blit=False)

    plt.tight_layout()
    plt.close()  # Don't display immediately

    return HTML(anim.to_jshtml())


def visualize_convex_hull(points, hull_points=None, figsize=(10, 8)):
    """
    Visualize the convex hull of a set of points.

    Args:
        points: List of points
        hull_points: List of points forming the convex hull (computed if not provided)
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)

    # Extract x and y coordinates
    x = [p.x for p in points]
    y = [p.y for p in points]

    # Plot all points
    plt.scatter(x, y, color='blue', label='Points')

    if hull_points is not None:
        # Extract hull coordinates
        hull_x = [p.x for p in hull_points]
        hull_y = [p.y for p in hull_points]

        # Close the hull
        hull_x.append(hull_points[0].x)
        hull_y.append(hull_points[0].y)

        # Plot the hull
        plt.plot(hull_x, hull_y, 'r-', linewidth=2, label='Convex Hull')
        plt.scatter(hull_x[:-1], hull_y[:-1], color='red')

    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.title('Convex Hull')
    plt.show()


def visualize_convex_hull_algorithm(algorithm: Callable, points, title="Convex Hull Algorithm",
                                    interval=500, figsize=(10, 8)):
    """
    Visualize a convex hull algorithm step by step.

    Args:
        algorithm: The convex hull algorithm function
        points: List of points
        title: Title for the animation
        interval: Time interval between frames in milliseconds
        figsize: Figure size as (width, height) tuple

    Returns:
        The animation object
    """
    # Setup history tracking
    history = []

    # Wrapper function to track steps
    def track_hull_steps(points):
        # Initial state - all points
        history.append((points, []))

        # Call the actual algorithm - different algorithm have different step tracking
        if algorithm.__name__ == 'graham_scan':
            # Sort points by polar angle
            p0 = min(points, key=lambda p: (p.y, p.x))
            sorted_points = sorted(points, key=lambda p: (
                math.atan2(p.y - p0.y, p.x - p0.x),
                (p.x - p0.x) ** 2 + (p.y - p0.y) ** 2
            ))
            history.append((points, sorted_points[:1]))

            # Process points
            hull = sorted_points[:3]
            history.append((points, hull.copy()))

            for i in range(3, len(sorted_points)):
                while len(hull) > 1:
                    # Check if adding this point creates a non-left turn
                    x1, y1 = hull[-2].x - hull[-1].x, hull[-2].y - hull[-1].y
                    x2, y2 = sorted_points[i].x - hull[-1].x, sorted_points[i].y - hull[-1].y
                    cross_product = x1 * y2 - y1 * x2

                    if cross_product > 0:  # Left turn
                        break

                    hull.pop()
                    history.append((points, hull.copy()))

                hull.append(sorted_points[i])
                history.append((points, hull.copy()))

            return hull
        else:
            # Generic approach for other algorithms
            result = algorithm(points)
            history.append((points, result))
            return result

    # Run the algorithm with tracking
    track_hull_steps(points)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.grid(True)

    # Extract coordinates
    x = [p.x for p in points]
    y = [p.y for p in points]

    # Set axis limits
    margin = 0.1 * (max(x) - min(x) + max(y) - min(y))
    ax.set_xlim(min(x) - margin, max(x) + margin)
    ax.set_ylim(min(y) - margin, max(y) + margin)

    # Plot all points
    ax.scatter(x, y, color='blue', label='Points')

    # Initialize hull line
    hull_line, = ax.plot([], [], 'r-', linewidth=2, label='Current Hull')
    hull_points_plot = ax.scatter([], [], color='red', label='Hull Points')

    # Add info text
    info_text = ax.text(0.5, 0.02, '', transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=12)

    # Initialize legend
    ax.legend()

    # Update function for animation
    def update(frame):
        points_all, current_hull = history[frame]

        if current_hull:
            # Extract hull coordinates
            hull_x = [p.x for p in current_hull]
            hull_y = [p.y for p in current_hull]

            # Close the hull for display if it has at least 3 points
            if len(current_hull) >= 3:
                hull_x.append(current_hull[0].x)
                hull_y.append(current_hull[0].y)

            # Update hull line and points
            hull_line.set_data(hull_x, hull_y)
            hull_points_plot.set_offsets(np.column_stack((hull_x[:-1] if len(current_hull) >= 3 else hull_x,
                                                          hull_y[:-1] if len(current_hull) >= 3 else hull_y)))
        else:
            hull_line.set_data([], [])
            hull_points_plot.set_offsets(np.zeros((0, 2)))

        # Update info text
        if frame == 0:
            info_text.set_text('Initial points')
        elif frame == len(history) - 1:
            info_text.set_text('Final convex hull')
        else:
            info_text.set_text(f'Step {frame}: Building hull with {len(current_hull)} points')

        return hull_line, hull_points_plot, info_text

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=interval, repeat=False, blit=True)

    plt.close()  # Don't display immediately

    return HTML(anim.to_jshtml())


def visualize_graph_algorithm(algorithm: Callable, graph, source=None, target=None,
                              title: str = None, layout='spring', interval=1000, figsize=(10, 8)):
    """
    Visualize a graph algorithm step by step.

    Args:
        algorithm: The graph algorithm function
        graph: The graph to visualize
        source: Source vertex for algorithms like BFS, DFS, Dijkstra
        target: Target vertex for pathfinding algorithms
        title: Title for the animation
        layout: Graph layout algorithm
        interval: Time interval between frames in milliseconds
        figsize: Figure size as (width, height) tuple

    Returns:
        The animation object
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required for graph visualization. Install with: pip install networkx")
        return

    # Setup NetworkX graph
    if hasattr(graph, 'directed') and graph.directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add nodes
    for vertex in graph.vertices:
        G.add_node(vertex)

    # Add edges with weights if available
    if hasattr(graph, 'weighted') and graph.weighted:
        for (u, v), weight in graph.edges.items():
            G.add_edge(u, v, weight=weight)
    else:
        for u, v in graph.edges:
            G.add_edge(u, v)

    # Compute layout once
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Setup history tracking
    history = []

    # Track node states at each step
    node_states = {
        'not_visited': [v for v in graph.vertices],
        'in_queue': [],
        'visiting': [],
        'visited': [],
        'current_path': []
    }

    # For shortest path tracking
    distances = {}
    predecessors = {}

    # Wrapper function to track steps
    def track_graph_steps():
        # Initial state
        history.append({
            'node_states': {k: v.copy() for k, v in node_states.items()},
            'distances': distances.copy(),
            'predecessors': predecessors.copy(),
            'info': 'Initial state'
        })

        # BFS instrumentation
        if algorithm.__name__ == 'breadth_first_search':
            if source is None:
                raise ValueError("Source vertex is required for BFS")

            queue = deque([source])
            node_states['not_visited'].remove(source)
            node_states['in_queue'].append(source)
            history.append({
                'node_states': {k: v.copy() for k, v in node_states.items()},
                'distances': distances.copy(),
                'predecessors': predecessors.copy(),
                'info': f'Added source {source} to queue'
            })

            visited = set([source])
            predecessors[source] = None

            while queue:
                vertex = queue.popleft()
                node_states['in_queue'].remove(vertex)
                node_states['visiting'].append(vertex)
                history.append({
                    'node_states': {k: v.copy() for k, v in node_states.items()},
                    'distances': distances.copy(),
                    'predecessors': predecessors.copy(),
                    'info': f'Processing vertex {vertex}'
                })

                for neighbor in graph.adjacency_list.get(vertex, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
                        predecessors[neighbor] = vertex

                        node_states['not_visited'].remove(neighbor)
                        node_states['in_queue'].append(neighbor)
                        history.append({
                            'node_states': {k: v.copy() for k, v in node_states.items()},
                            'distances': distances.copy(),
                            'predecessors': predecessors.copy(),
                            'info': f'Added neighbor {neighbor} to queue'
                        })

                node_states['visiting'].remove(vertex)
                node_states['visited'].append(vertex)
                history.append({
                    'node_states': {k: v.copy() for k, v in node_states.items()},
                    'distances': distances.copy(),
                    'predecessors': predecessors.copy(),
                    'info': f'Finished processing {vertex}'
                })

            # Construct path if target is specified
            if target is not None and target in predecessors:
                path = []
                current = target
                while current is not None:
                    path.append(current)
                    current = predecessors.get(current)
                path.reverse()

                node_states['current_path'] = path
                history.append({
                    'node_states': {k: v.copy() for k, v in node_states.items()},
                    'distances': distances.copy(),
                    'predecessors': predecessors.copy(),
                    'info': f'Final path: {" -> ".join(str(v) for v in path)}'
                })

            return predecessors

        # DFS instrumentation
        elif algorithm.__name__ == 'depth_first_search':
            if source is None:
                raise ValueError("Source vertex is required for DFS")

            visited = {}

            def dfs_visit(vertex, parent=None):
                visited[vertex] = True
                node_states['not_visited'].remove(vertex)
                node_states['visiting'].append(vertex)
                predecessors[vertex] = parent
                history.append({
                    'node_states': {k: v.copy() for k, v in node_states.items()},
                    'distances': distances.copy(),
                    'predecessors': predecessors.copy(),
                    'info': f'Visiting {vertex}'
                })

                for neighbor in graph.adjacency_list.get(vertex, []):
                    if neighbor not in visited:
                        dfs_visit(neighbor, vertex)

                node_states['visiting'].remove(vertex)
                node_states['visited'].append(vertex)
                history.append({
                    'node_states': {k: v.copy() for k, v in node_states.items()},
                    'distances': distances.copy(),
                    'predecessors': predecessors.copy(),
                    'info': f'Finished with {vertex}'
                })

            dfs_visit(source)

            # Construct path if target is specified
            if target is not None and target in predecessors:
                path = []
                current = target
                while current is not None:
                    path.append(current)
                    current = predecessors.get(current)
                path.reverse()

                node_states['current_path'] = path
                history.append({
                    'node_states': {k: v.copy() for k, v in node_states.items()},
                    'distances': distances.copy(),
                    'predecessors': predecessors.copy(),
                    'info': f'Final path: {" -> ".join(str(v) for v in path)}'
                })

            return visited

        # Dijkstra instrumentation
        elif algorithm.__name__ == 'dijkstra':
            if source is None:
                raise ValueError("Source vertex is required for Dijkstra's algorithm")

            import heapq

            # Initialize distances
            for vertex in graph.vertices:
                distances[vertex] = float('inf')
                predecessors[vertex] = None

            distances[source] = 0
            pq = [(0, source)]  # (distance, vertex)

            node_states['not_visited'].remove(source)
            node_states['in_queue'].append(source)
            history.append({
                'node_states': {k: v.copy() for k, v in node_states.items()},
                'distances': distances.copy(),
                'predecessors': predecessors.copy(),
                'info': f'Added source {source} to priority queue with distance 0'
            })

            while pq:
                dist, vertex = heapq.heappop(pq)

                # Skip outdated entries
                if dist > distances[vertex]:
                    continue

                node_states['in_queue'].remove(vertex)
                node_states['visiting'].append(vertex)
                history.append({
                    'node_states': {k: v.copy() for k, v in node_states.items()},
                    'distances': distances.copy(),
                    'predecessors': predecessors.copy(),
                    'info': f'Processing vertex {vertex} with distance {dist}'
                })

                # Process neighbors
                for neighbor, weight in graph.get_neighbors_with_weights(vertex).items():
                    new_dist = dist + weight

                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        predecessors[neighbor] = vertex
                        heapq.heappush(pq, (new_dist, neighbor))

                        if neighbor in node_states['not_visited']:
                            node_states['not_visited'].remove(neighbor)
                            node_states['in_queue'].append(neighbor)
                        elif neighbor in node_states['visited']:
                            node_states['visited'].remove(neighbor)
                            node_states['in_queue'].append(neighbor)

                        history.append({
                            'node_states': {k: v.copy() for k, v in node_states.items()},
                            'distances': distances.copy(),
                            'predecessors': predecessors.copy(),
                            'info': f'Updated distance to {neighbor}: {new_dist}'
                        })

                node_states['visiting'].remove(vertex)
                node_states['visited'].append(vertex)
                history.append({
                    'node_states': {k: v.copy() for k, v in node_states.items()},
                    'distances': distances.copy(),
                    'predecessors': predecessors.copy(),
                    'info': f'Finished processing {vertex}'
                })

            # Construct path if target is specified
            if target is not None:
                if distances[target] == float('inf'):
                    history.append({
                        'node_states': {k: v.copy() for k, v in node_states.items()},
                        'distances': distances.copy(),
                        'predecessors': predecessors.copy(),
                        'info': f'No path exists to target {target}'
                    })
                else:
                    path = []
                    current = target
                    while current is not None:
                        path.append(current)
                        current = predecessors.get(current)
                    path.reverse()

                    node_states['current_path'] = path
                    history.append({
                        'node_states': {k: v.copy() for k, v in node_states.items()},
                        'distances': distances.copy(),
                        'predecessors': predecessors.copy(),
                        'info': f'Final path to {target} (distance {distances[target]}): {" -> ".join(str(v) for v in path)}'
                    })

            return distances, predecessors

        else:
            # Generic approach for other algorithms
            result = algorithm(graph, source, target) if source else algorithm(graph)
            history.append({
                'node_states': {k: v.copy() for k, v in node_states.items()},
                'distances': {},
                'predecessors': {},
                'info': 'Algorithm completed'
            })
            return result

    # Run the algorithm with tracking
    track_graph_steps()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)

    # Node color mapping
    node_colors = {
        'not_visited': 'lightblue',
        'in_queue': 'yellow',
        'visiting': 'orange',
        'visited': 'green',
        'current_path': 'red'
    }

    # Initialize node and edge collections
    node_collections = {}
    for state, color in node_colors.items():
        node_collections[state] = nx.draw_networkx_nodes(
            G, pos, nodelist=[], node_color=color, node_size=500, alpha=0.8
        )

    # Draw all edges
    nx.draw_networkx_edges(
        G, pos, width=1.5,
        arrowstyle='-|>' if hasattr(graph, 'directed') and graph.directed else '-',
        arrowsize=15
    )

    # Draw edge labels if weighted
    if hasattr(graph, 'weighted') and graph.weighted:
        edge_labels = {(u, v): str(w) for (u, v), w in graph.edges.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Add info text
    info_text = ax.text(0.5, 0.02, '', transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7))

    # Create a separate box for distances if using Dijkstra
    if algorithm.__name__ == 'dijkstra':
        distance_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                ha='left', va='top', fontsize=10,
                                bbox=dict(facecolor='white', alpha=0.7))
    else:
        distance_text = None

    # Update function for animation
    def update(frame):
        frame_data = history[frame]

        # Update node colors
        for state, nodes in frame_data['node_states'].items():
            node_collections[state].set_nodelist(nodes)

        # Update info text
        info_text.set_text(frame_data['info'])

        # Update distance text for Dijkstra
        if distance_text is not None and frame_data['distances']:
            distances_str = "\n".join([
                f"{v}: {dist if dist != float('inf') else 'inf'}"
                for v, dist in sorted(frame_data['distances'].items())
            ])
            distance_text.set_text(f"Distances:\n{distances_str}")

        elements = list(node_collections.values()) + [info_text]
        if distance_text:
            elements.append(distance_text)

        return elements

    # Turn off axis
    ax.axis('off')

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=interval, repeat=False, blit=False)

    plt.close()  # Don't display immediately

    return HTML(anim.to_jshtml())


def visualize_binary_search(arr: List[int], target: int, interval=500, figsize=(10, 6)):
    """
    Visualize binary search algorithm step by step.

    Args:
        arr: Sorted array to search in
        target: Value to search for
        interval: Time interval between frames in milliseconds
        figsize: Figure size as (width, height) tuple

    Returns:
        The animation object
    """
    # Make sure the array is sorted
    if not all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)):
        arr = sorted(arr)

    # Setup history tracking
    history = []

    # Binary search with tracking
    def binary_search_with_tracking(arr, target):
        history.append({
            'left': 0,
            'right': len(arr) - 1,
            'mid': None,
            'found': False,
            'info': 'Starting binary search'
        })

        left, right = 0, len(arr) - 1

        while left <= right:
            mid = (left + right) // 2
            history.append({
                'left': left,
                'right': right,
                'mid': mid,
                'found': arr[mid] == target,
                'info': f'Checking index {mid}, value {arr[mid]}'
            })

            if arr[mid] == target:
                history.append({
                    'left': left,
                    'right': right,
                    'mid': mid,
                    'found': True,
                    'info': f'Found target {target} at index {mid}'
                })
                return mid

            if arr[mid] < target:
                left = mid + 1
                history.append({
                    'left': left,
                    'right': right,
                    'mid': None,
                    'found': False,
                    'info': f'{arr[mid]} < {target}, searching right half: [{left}...{right}]'
                })
            else:
                right = mid - 1
                history.append({
                    'left': left,
                    'right': right,
                    'mid': None,
                    'found': False,
                    'info': f'{arr[mid]} > {target}, searching left half: [{left}...{right}]'
                })

        history.append({
            'left': left,
            'right': right,
            'mid': None,
            'found': False,
            'info': f'Target {target} not found in array'
        })

        return -1

    # Run the search with tracking
    binary_search_with_tracking(arr, target)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Binary Search for {target} in Sorted Array')

    # Create initial bar chart
    bars = ax.bar(range(len(arr)), arr, color='lightblue', edgecolor='black')
    ax.set_xticks(range(len(arr)))
    ax.set_xticklabels([str(i) for i in range(len(arr))])

    # Set y-axis limit with some padding
    ax.set_ylim(0, max(arr) * 1.2)

    # Add target line
    target_line = ax.axhline(y=target, color='red', linestyle='--', alpha=0.7, label=f'Target: {target}')

    # Add info text
    info_text = ax.text(0.5, 0.95, '', transform=ax.transAxes,
                        ha='center', va='top', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7))

    # Update function for animation
    def update(frame):
        frame_data = history[frame]

        # Reset all bars to default color
        for bar in bars:
            bar.set_color('lightblue')
            bar.set_edgecolor('black')

        # Color the current search range
        left, right = frame_data['left'], frame_data['right']
        for i in range(left, right + 1):
            if 0 <= i < len(arr):
                bars[i].set_color('skyblue')
                bars[i].set_edgecolor('blue')

        # Highlight the middle element
        if frame_data['mid'] is not None:
            mid = frame_data['mid']
            bars[mid].set_color('yellow' if not frame_data['found'] else 'green')
            bars[mid].set_edgecolor('black')

        # Update info text
        info_text.set_text(frame_data['info'])

        return bars + [info_text, target_line]

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=interval, repeat=False, blit=False)

    plt.legend()
    plt.tight_layout()
    plt.close()  # Don't display immediately

    return HTML(anim.to_jshtml())


def visualize_tree_traversal(root, traversal_type='inorder', interval=500, figsize=(10, 8)):
    """
    Visualize tree traversal algorithms step by step.

    Args:
        root: Root node of the binary tree
        traversal_type: Type of traversal ('inorder', 'preorder', 'postorder', 'levelorder')
        interval: Time interval between frames in milliseconds
        figsize: Figure size as (width, height) tuple

    Returns:
        The animation object
    """
    if root is None:
        print("Empty tree")
        return

    # Calculate tree dimensions
    def height(node):
        if node is None:
            return 0
        return 1 + max(height(node.left), height(node.right))

    def width(node):
        if node is None:
            return 0
        return 1 + width(node.left) + width(node.right)

    tree_height = height(root)
    tree_width = width(root)

    # Setup history tracking
    history = []

    # Traversal algorithms with tracking
    def inorder_traversal(node, path=[]):
        if node:
            # Recursively traverse left subtree
            history.append({
                'current': node,
                'visited': path.copy(),
                'status': 'Going to left subtree',
                'action': 'down_left'
            })
            inorder_traversal(node.left, path)

            # Visit current node
            path.append(node)
            history.append({
                'current': node,
                'visited': path.copy(),
                'status': f'Visiting node {node.value}',
                'action': 'visit'
            })

            # Recursively traverse right subtree
            history.append({
                'current': node,
                'visited': path.copy(),
                'status': 'Going to right subtree',
                'action': 'down_right'
            })
            inorder_traversal(node.right, path)

    def preorder_traversal(node, path=[]):
        if node:
            # Visit current node
            path.append(node)
            history.append({
                'current': node,
                'visited': path.copy(),
                'status': f'Visiting node {node.value}',
                'action': 'visit'
            })

            # Recursively traverse left subtree
            history.append({
                'current': node,
                'visited': path.copy(),
                'status': 'Going to left subtree',
                'action': 'down_left'
            })
            preorder_traversal(node.left, path)

            # Recursively traverse right subtree
            history.append({
                'current': node,
                'visited': path.copy(),
                'status': 'Going to right subtree',
                'action': 'down_right'
            })
            preorder_traversal(node.right, path)

    def postorder_traversal(node, path=[]):
        if node:
            # Recursively traverse left subtree
            history.append({
                'current': node,
                'visited': path.copy(),
                'status': 'Going to left subtree',
                'action': 'down_left'
            })
            postorder_traversal(node.left, path)

            # Recursively traverse right subtree
            history.append({
                'current': node,
                'visited': path.copy(),
                'status': 'Going to right subtree',
                'action': 'down_right'
            })
            postorder_traversal(node.right, path)

            # Visit current node
            path.append(node)
            history.append({
                'current': node,
                'visited': path.copy(),
                'status': f'Visiting node {node.value}',
                'action': 'visit'
            })

    def levelorder_traversal(root, path=[]):
        if not root:
            return

        queue = deque([root])

        while queue:
            node = queue.popleft()
            path.append(node)

            history.append({
                'current': node,
                'visited': path.copy(),
                'status': f'Visiting node {node.value}',
                'action': 'visit'
            })

            if node.left:
                queue.append(node.left)
                history.append({
                    'current': node,
                    'visited': path.copy(),
                    'status': f'Adding left child {node.left.value} to queue',
                    'action': 'queue_left'
                })

            if node.right:
                queue.append(node.right)
                history.append({
                    'current': node,
                    'visited': path.copy(),
                    'status': f'Adding right child {node.right.value} to queue',
                    'action': 'queue_right'
                })

    # Initial history entry
    history.append({
        'current': root,
        'visited': [],
        'status': f'Starting {traversal_type} traversal',
        'action': 'start'
    })

    # Run the traversal with tracking
    if traversal_type == 'inorder':
        inorder_traversal(root, [])
    elif traversal_type == 'preorder':
        preorder_traversal(root, [])
    elif traversal_type == 'postorder':
        postorder_traversal(root, [])
    elif traversal_type == 'levelorder':
        levelorder_traversal(root, [])
    else:
        raise ValueError(f"Unknown traversal type: {traversal_type}")

    # Final history entry
    history.append({
        'current': None,
        'visited': history[-1]['visited'],
        'status': f'Traversal complete: {", ".join(str(node.value) for node in history[-1]["visited"])}',
        'action': 'complete'
    })

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'{traversal_type.capitalize()} Traversal')

    # Define drawing functions
    def draw_tree(highlight_node=None, visited_nodes=None):
        ax.clear()
        ax.set_xlim(-tree_width, tree_width)
        ax.set_ylim(-tree_height * 2, 2)
        ax.axis('off')

        # Set node colors
        node_colors = {
            'default': 'lightblue',
            'highlight': 'yellow',
            'visited': 'lightgreen'
        }

        # Draw the tree recursively
        def draw_node(node, x, y, dx, edges=None):
            if node is None:
                return

            if edges is None:
                edges = []

            # Determine node color
            if node == highlight_node:
                color = node_colors['highlight']
            elif visited_nodes and node in visited_nodes:
                color = node_colors['visited']
            else:
                color = node_colors['default']

            # Draw the node
            circle = plt.Circle((x, y), 0.5, fill=True, color=color, ec='black')
            ax.add_patch(circle)

            # Add the node value
            ax.text(x, y, str(node.value), ha='center', va='center')

            # Calculate new_dx (half the horizontal distance to the next node)
            new_dx = dx / 2

            # Draw left child and edge if it exists
            if node.left:
                left_x = x - dx
                left_y = y - 2
                edges.append(((x, y - 0.5), (left_x, left_y + 0.5)))
                draw_node(node.left, left_x, left_y, new_dx, edges)

            # Draw right child and edge if it exists
            if node.right:
                right_x = x + dx
                right_y = y - 2
                edges.append(((x, y - 0.5), (right_x, right_y + 0.5)))
                draw_node(node.right, right_x, right_y, new_dx, edges)

            return edges

        # Draw the tree
        edges = draw_node(root, 0, 0, 2 ** (tree_height - 1))

        # Draw edges
        if edges:
            for (x1, y1), (x2, y2) in edges:
                ax.plot([x1, x2], [y1, y2], 'k-')

        # Add traversal order so far
        if visited_nodes:
            traversal_text = ", ".join(str(node.value) for node in visited_nodes)
            ax.text(0.5, -0.05, f'Traversal so far: {traversal_text}',
                    transform=ax.transAxes, ha='center', va='bottom')

        # Add info box
        if 'status' in history[0]:
            status_text = history[0]['status']
            ax.text(0.5, 1.05, status_text, transform=ax.transAxes,
                    ha='center', va='top', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))

    # Draw initial tree
    draw_tree()

    # Update function for animation
    def update(frame):
        frame_data = history[frame]
        draw_tree(highlight_node=frame_data['current'], visited_nodes=frame_data['visited'])

        # No artists to return because we redraw the whole figure
        return []

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=interval, repeat=False, blit=False)

    plt.tight_layout()
    plt.close()  # Don't display immediately

    return HTML(anim.to_jshtml())


def visualize_dynamic_programming(algorithm: Callable, problem_instance: Any,
                                  title: str = None, interval=500, figsize=(10, 6)):
    """
    Visualize dynamic programming algorithms step by step.

    Args:
        algorithm: The DP algorithm function
        problem_instance: The problem instance to solve
        title: Title for the animation
        interval: Time interval between frames in milliseconds
        figsize: Figure size as (width, height) tuple

    Returns:
        The animation object
    """
    # Setup history tracking
    history = []

    # Wrap the algorithm to track steps
    def track_dp_steps(problem_instance):
        # Different instrumentation for different DP algorithms
        if algorithm.__name__ == 'fibonacci_dp':
            n = problem_instance
            dp = [0] * (n + 1)

            # Base cases
            if n >= 0:
                dp[0] = 0
                history.append({
                    'dp_table': dp.copy(),
                    'current_index': 0,
                    'info': 'Base case: F(0) = 0'
                })

            if n >= 1:
                dp[1] = 1
                history.append({
                    'dp_table': dp.copy(),
                    'current_index': 1,
                    'info': 'Base case: F(1) = 1'
                })

            # Fill the DP table
            for i in range(2, n + 1):
                dp[i] = dp[i - 1] + dp[i - 2]
                history.append({
                    'dp_table': dp.copy(),
                    'current_index': i,
                    'info': f'F({i}) = F({i - 1}) + F({i - 2}) = {dp[i - 1]} + {dp[i - 2]} = {dp[i]}'
                })

            return dp[n]

        elif algorithm.__name__ == 'knapsack_01':
            values, weights, capacity = problem_instance
            n = len(values)

            # Create DP table
            dp = [[0] * (capacity + 1) for _ in range(n + 1)]

            # Base case - already initialized to 0
            history.append({
                'dp_table': [row.copy() for row in dp],
                'current_cell': None,
                'info': 'Base case: Empty knapsack or no items'
            })

            # Fill the DP table
            for i in range(1, n + 1):
                for w in range(capacity + 1):
                    if weights[i - 1] <= w:
                        # Either take the item or don't take it
                        take = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                        dont_take = dp[i - 1][w]
                        dp[i][w] = max(take, dont_take)

                        history.append({
                            'dp_table': [row.copy() for row in dp],
                            'current_cell': (i, w),
                            'info': f'Item {i} (value={values[i - 1]}, weight={weights[i - 1]}): '
                                    f'Max of take ({take}) or don\'t take ({dont_take})'
                        })
                    else:
                        # Can't take the item because it's too heavy
                        dp[i][w] = dp[i - 1][w]

                        history.append({
                            'dp_table': [row.copy() for row in dp],
                            'current_cell': (i, w),
                            'info': f'Item {i} (weight={weights[i - 1]}) too heavy for capacity {w}: '
                                    f'dp[{i}][{w}] = dp[{i - 1}][{w}] = {dp[i - 1][w]}'
                        })

            return dp[n][capacity]

        elif algorithm.__name__ == 'longest_common_subsequence':
            s1, s2 = problem_instance
            m, n = len(s1), len(s2)

            # Create DP table
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            # Base case - already initialized to 0
            history.append({
                'dp_table': [row.copy() for row in dp],
                'current_cell': None,
                'info': 'Base case: Empty strings'
            })

            # Fill the DP table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        history.append({
                            'dp_table': [row.copy() for row in dp],
                            'current_cell': (i, j),
                            'info': f'Characters match: {s1[i - 1]} == {s2[j - 1]}, '
                                    f'dp[{i}][{j}] = dp[{i - 1}][{j - 1}] + 1 = {dp[i - 1][j - 1]} + 1 = {dp[i][j]}'
                        })
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                        history.append({
                            'dp_table': [row.copy() for row in dp],
                            'current_cell': (i, j),
                            'info': f'Characters don\'t match: {s1[i - 1]} != {s2[j - 1]}, '
                                    f'dp[{i}][{j}] = max(dp[{i - 1}][{j}], dp[{i}][{j - 1}]) = '
                                    f'max({dp[i - 1][j]}, {dp[i][j - 1]}) = {dp[i][j]}'
                        })

            # Final state
            history.append({
                'dp_table': [row.copy() for row in dp],
                'current_cell': None,
                'info': f'Final DP table. LCS length: {dp[m][n]}'
            })

            # Reconstruct the LCS
            lcs = []
            i, j = m, n
            while i > 0 and j > 0:
                if s1[i - 1] == s2[j - 1]:
                    lcs.append(s1[i - 1])
                    i -= 1
                    j -= 1
                    history.append({
                        'dp_table': [row.copy() for row in dp],
                        'current_cell': (i + 1, j + 1),
                        'lcs_so_far': ''.join(reversed(lcs)),
                        'info': f'Characters match. Add {s1[i]} to LCS. Move diagonally.'
                    })
                elif dp[i - 1][j] > dp[i][j - 1]:
                    i -= 1
                    history.append({
                        'dp_table': [row.copy() for row in dp],
                        'current_cell': (i + 1, j),
                        'lcs_so_far': ''.join(reversed(lcs)),
                        'info': 'Move up.'
                    })
                else:
                    j -= 1
                    history.append({
                        'dp_table': [row.copy() for row in dp],
                        'current_cell': (i, j + 1),
                        'lcs_so_far': ''.join(reversed(lcs)),
                        'info': 'Move left.'
                    })

            # Final result
            lcs_result = ''.join(reversed(lcs))
            history.append({
                'dp_table': [row.copy() for row in dp],
                'current_cell': None,
                'lcs_so_far': lcs_result,
                'info': f'LCS reconstruction complete: {lcs_result}'
            })

            return lcs_result

        else:
            # Generic approach for other algorithms
            result = algorithm(problem_instance)
            history.append({
                'result': result,
                'info': 'Algorithm completed'
            })
            return result

    # Run the algorithm with tracking
    track_dp_steps(problem_instance)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)

    # Draw initial DP table
    def plot_dp_table(frame_data):
        ax.clear()

        # Set title
        if title:
            ax.set_title(title)

        # Different visualization for different DP problems
        if 'dp_table' in frame_data:
            dp_table = frame_data['dp_table']

            if isinstance(dp_table[0], list):
                # 2D DP table
                m, n = len(dp_table), len(dp_table[0])

                # Create table
                table = ax.imshow(dp_table, cmap='Blues', aspect='auto')
                plt.colorbar(table, ax=ax, label='Value')

                # Add grid
                ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

                # Add row and column labels
                ax.set_xticks(np.arange(n))
                ax.set_yticks(np.arange(m))

                # Add values in cells
                for i in range(m):
                    for j in range(n):
                        ax.text(j, i, str(dp_table[i][j]), ha='center', va='center',
                                color='black' if dp_table[i][j] < max(map(max, dp_table)) / 2 else 'white')

                # Highlight current cell
                if 'current_cell' in frame_data and frame_data['current_cell'] is not None:
                    i, j = frame_data['current_cell']
                    if 0 <= i < m and 0 <= j < n:
                        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
                        ax.add_patch(rect)

                # Add LCS if available
                if 'lcs_so_far' in frame_data:
                    ax.text(0.5, -0.1, f"LCS: {frame_data['lcs_so_far']}", transform=ax.transAxes,
                            ha='center', va='center', fontsize=12,
                            bbox=dict(facecolor='lightgreen', alpha=0.7))
            else:
                # 1D DP table
                n = len(dp_table)

                # Plot as a bar chart
                bars = ax.bar(range(n), dp_table, color='skyblue', edgecolor='black')

                # Highlight current index
                if 'current_index' in frame_data and frame_data['current_index'] is not None:
                    i = frame_data['current_index']
                    if 0 <= i < n:
                        bars[i].set_color('yellow')
                        bars[i].set_edgecolor('red')

                # Set labels
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.set_xticks(range(n))

        # Add info text
        if 'info' in frame_data:
            ax.text(0.5, 1.05, frame_data['info'], transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))

    # Draw initial frame
    plot_dp_table(history[0])

    # Update function for animation
    def update(frame):
        frame_data = history[frame]
        plot_dp_table(frame_data)
        return []

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=interval, repeat=False, blit=False)

    plt.tight_layout()
    plt.close()  # Don't display immediately

    return HTML(anim.to_jshtml())

def visualize_fitness_history(fitness_history: List[float],
                              best_fitness_history: Optional[List[float]] = None,
                              title: str = "Fitness Over Generations",
                              figsize: Tuple[int, int] = (10, 6)):
    """
    Visualize the fitness evolution over generations in a genetic algorithm.

    Args:
        fitness_history: List of average fitness values per generation
        best_fitness_history: Optional list of the best fitness values per generation
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)
    generations = range(1, len(fitness_history) + 1)

    plt.plot(generations, fitness_history, 'b-', label='Average Fitness')

    if best_fitness_history is not None:
        plt.plot(generations, best_fitness_history, 'r-', label='Best Fitness')

    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def visualize_population_diversity(population: List[Any],
                                   distance_function: Callable[[Any, Any], float],
                                   generation: int = 0,
                                   title: str = "Population Diversity",
                                   figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize the diversity of a population in a genetic algorithm using MDS.

    Args:
        population: List of individuals in the population
        distance_function: Function to calculate distance between two individuals
        generation: Current generation number
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    try:
        from sklearn.manifold import MDS
    except ImportError:
        print("scikit-learn is required for diversity visualization. Install with: pip install scikit-learn")
        return

    # Calculate pairwise distances between individuals
    n = len(population)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = distance_function(population[i], population[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    # Apply MDS to project individuals to 2D space
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(distance_matrix)

    # Visualize the population
    plt.figure(figsize=figsize)

    plt.scatter(positions[:, 0], positions[:, 1], alpha=0.7)

    plt.title(f"{title} (Generation {generation})")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_tsp_solution(distance_matrix: List[List[float]],
                           tour: List[int],
                           title: str = "TSP Solution",
                           figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize a solution to the Traveling Salesman Problem.

    Args:
        distance_matrix: Matrix of distances between cities
        tour: List of city indices representing the tour
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    n = len(distance_matrix)

    # Create a fully connected graph
    G = nx.Graph()

    # Generate node positions (in a circle by default)
    pos = {}
    for i in range(n):
        angle = 2 * math.pi * i / n
        pos[i] = (math.cos(angle), math.sin(angle))

    # Add nodes
    for i in range(n):
        G.add_node(i)

    # Add all edges with their weights
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=distance_matrix[i][j])

    # Create figure
    plt.figure(figsize=figsize)

    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

    # Highlight the tour edges
    tour_edges = [(tour[i], tour[(i + 1) % n]) for i in range(n)]
    nx.draw_networkx_edges(G, pos, edgelist=tour_edges, width=2, edge_color='red')

    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Calculate the tour length
    tour_length = sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))

    plt.title(f"{title}\nTour Length: {tour_length:.2f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def animate_genetic_algorithm(ga_instance, problem_instance,
                              iterations_per_frame: int = 1,
                              max_frames: int = 100,
                              figsize: Tuple[int, int] = (10, 6),
                              visualize_solution: Callable = None):
    """
    Create an animation of a genetic algorithm's progress.

    Args:
        ga_instance: Instance of a genetic algorithm class
        problem_instance: The problem to solve
        iterations_per_frame: Number of generations to run between frames
        max_frames: Maximum number of frames in the animation
        figsize: Figure size as (width, height) tuple
        visualize_solution: Function to visualize the current best solution

    Returns:
        Animation HTML that can be displayed in a notebook
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Store history
    fitness_history = []
    best_fitness_history = []
    best_individuals = []

    # Set up the run function that will be called to make each frame
    def run_generations(frame):
        nonlocal ga_instance, problem_instance

        # Run the algorithm for a few iterations
        for _ in range(iterations_per_frame):
            # Extract the current state
            population = ga_instance.population
            evaluated_population = ga_instance.evaluate_population(population)

            # Get best and average fitness
            current_best = max(evaluated_population, key=lambda x: x[1])
            avg_fitness = sum(fitness for _, fitness in evaluated_population) / len(evaluated_population)

            # Store history
            fitness_history.append(avg_fitness)
            best_fitness_history.append(current_best[1])
            best_individuals.append(current_best[0])

            # Create next generation
            population = ga_instance.create_next_generation(evaluated_population)
            ga_instance.population = population

        # Clear the axes for new frame
        ax1.clear()
        ax2.clear()

        # Plot fitness history
        generations = range(1, len(fitness_history) + 1)
        ax1.plot(generations, fitness_history, 'b-', label='Average Fitness')
        ax1.plot(generations, best_fitness_history, 'r-', label='Best Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Visualize current best solution if a visualization function is provided
        if visualize_solution:
            visualize_solution(ax2, best_individuals[-1], problem_instance)
        else:
            ax2.text(0.5, 0.5, f"Best Fitness: {best_fitness_history[-1]:.4f}",
                     ha='center', va='center', fontsize=12)
            ax2.set_title('Current Best Solution')
            ax2.axis('off')

        # Return all artists that need to be updated
        return []

    # Create the animation
    anim = FuncAnimation(fig, run_generations, frames=max_frames,
                         interval=200, blit=False)

    plt.close(fig)  # Prevent display of the figure

    return HTML(anim.to_jshtml())


def animate_tsp_solution(ga_instance, distance_matrix,
                         iterations_per_frame: int = 1,
                         max_frames: int = 100,
                         figsize: Tuple[int, int] = (12, 6)):
    """
    Create an animation specifically for the TSP problem.

    Args:
        ga_instance: Instance of the TSPGeneticSolver
        distance_matrix: Matrix of distances between cities
        iterations_per_frame: Number of generations to run between frames
        max_frames: Maximum number of frames in the animation
        figsize: Figure size as (width, height) tuple

    Returns:
        Animation HTML that can be displayed in a notebook
    """

    # Create a visualization function for TSP
    def visualize_tsp(ax, tour, distance_matrix):
        n = len(distance_matrix)

        # Generate node positions (in a circle)
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n
            pos[i] = (math.cos(angle), math.sin(angle))

        # Clear the axis
        ax.clear()

        # Draw the nodes
        for i in range(n):
            ax.plot(pos[i][0], pos[i][1], 'o', markersize=10, color='lightblue')
            ax.text(pos[i][0] * 1.1, pos[i][1] * 1.1, str(i), fontsize=12)

        # Draw the tour edges
        for i in range(n):
            j = (i + 1) % n
            ax.plot([pos[tour[i]][0], pos[tour[j]][0]],
                    [pos[tour[i]][1], pos[tour[j]][1]], 'r-', linewidth=2)

        # Calculate tour length
        tour_length = sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))

        ax.set_title(f"TSP Tour (Length: {tour_length:.2f})")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('equal')
        ax.axis('off')

    return animate_genetic_algorithm(
        ga_instance=ga_instance,
        problem_instance=distance_matrix,
        iterations_per_frame=iterations_per_frame,
        max_frames=max_frames,
        figsize=figsize,
        visualize_solution=visualize_tsp
    )


def visualize_nsga2_front(solutions, objective_functions,
                          title: str = "Pareto Front",
                          figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize the Pareto front for a multi-objective optimization problem.

    Args:
        solutions: List of solutions (chromosomes, individuals)
        objective_functions: List of objective functions to evaluate
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    # Evaluate solutions
    objective_values = []
    for solution in solutions:
        values = [func(solution) for func in objective_functions]
        objective_values.append(values)

    # Convert to numpy array for easier manipulation
    objective_values = np.array(objective_values)

    # Create figure
    plt.figure(figsize=figsize)

    # For 2D objective space
    if len(objective_functions) == 2:
        plt.scatter(objective_values[:, 0], objective_values[:, 1], c='blue', alpha=0.7)
        plt.xlabel(f'Objective 1')
        plt.ylabel(f'Objective 2')

    # For 3D objective space
    elif len(objective_functions) == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(objective_values[:, 0], objective_values[:, 1], objective_values[:, 2],
                   c='blue', alpha=0.7)
        ax.set_xlabel(f'Objective 1')
        ax.set_ylabel(f'Objective 2')
        ax.set_zlabel(f'Objective 3')

    else:
        # For higher dimensions, use a parallel coordinates plot
        try:
            from pandas.plotting import parallel_coordinates
            import pandas as pd

            # Create a DataFrame
            df = pd.DataFrame(objective_values,
                              columns=[f'Objective {i + 1}' for i in range(len(objective_functions))])
            df['Solution'] = 'Pareto Optimal'

            # Normalize the values for better visualization
            for col in df.columns[:-1]:
                if df[col].max() != df[col].min():
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

            # Plot
            parallel_coordinates(df, 'Solution', colormap='viridis')

        except ImportError:
            plt.text(0.5, 0.5, f"Visualization for {len(objective_functions)} objectives requires pandas",
                     ha='center', va='center')

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def animate_nsga2(nsga2_instance, problem_instance,
                  iterations_per_frame: int = 1,
                  max_frames: int = 100,
                  figsize: Tuple[int, int] = (12, 6)):
    """
    Create an animation for the NSGA-II algorithm.

    Args:
        nsga2_instance: Instance of the NSGA2 class
        problem_instance: The problem to solve
        iterations_per_frame: Number of generations to run between frames
        max_frames: Maximum number of frames in the animation
        figsize: Figure size as (width, height) tuple

    Returns:
        Animation HTML that can be displayed in a notebook
    """
    # Create figure with two subplots (one for objectives, one for decision variables)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get dimensionality of the problem
    num_objectives = nsga2_instance.num_objectives

    # Store history
    fronts_history = []

    # Initialize population
    population = nsga2_instance.initialize_population()

    # Set up the run function that will be called to make each frame
    def run_generations(frame):
        nonlocal nsga2_instance, population

        # Run the algorithm for a few iterations
        for _ in range(iterations_per_frame):
            # Evaluate population
            evaluated_population = nsga2_instance.evaluate_population(population)

            # Get non-dominated fronts
            fronts = nsga2_instance.fast_non_dominated_sort(evaluated_population)

            # Store first front (Pareto optimal solutions)
            fronts_history.append([evaluated_population[i] for i in fronts[0]])

            # Calculate crowding distance
            crowding_distances = [
                nsga2_instance.crowding_distance_assignment(evaluated_population, front)
                for front in fronts
            ]

            # Select parents and create offspring
            parents = nsga2_instance.tournament_selection(evaluated_population, fronts, crowding_distances)
            offspring = nsga2_instance.create_offspring(parents)

            # Update population with the offspring
            population = offspring

        # Clear the axes for new frame
        ax1.clear()
        ax2.clear()

        # Get the latest Pareto front
        current_front = fronts_history[-1]

        # Extract objective values
        objective_values = [chrom[1] for chrom in current_front]

        # Plot objectives (2D case for simplicity)
        if num_objectives >= 2:
            ax1.scatter([v[0] for v in objective_values], [v[1] for v in objective_values],
                        c='blue', alpha=0.7)
            ax1.set_xlabel('Objective 1')
            ax1.set_ylabel('Objective 2')
            ax1.set_title('Objective Space (Pareto Front)')
            ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot decision variables (using parallel coordinates if available)
        try:
            from pandas.plotting import parallel_coordinates
            import pandas as pd

            # Get decision variables
            decision_vars = [chrom[0] for chrom in current_front]

            # Create a DataFrame
            df = pd.DataFrame(decision_vars,
                              columns=[f'Var {i + 1}' for i in range(len(decision_vars[0]))])
            df['Solution'] = 'Pareto Optimal'

            # Normalize for better visualization
            for col in df.columns[:-1]:
                if df[col].max() != df[col].min():
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

            # Plot
            parallel_coordinates(df, 'Solution', ax=ax2, colormap='viridis')
            ax2.set_title('Decision Variable Space')

        except (ImportError, ValueError):
            # Fallback visualization
            if len(current_front) > 0:
                ax2.text(0.5, 0.5, f"Front Size: {len(current_front)}\nGeneration: {len(fronts_history)}",
                         ha='center', va='center')
            ax2.set_title('Current Generation Information')
            ax2.axis('off')

        # Add generation number
        fig.suptitle(f'Generation {len(fronts_history)}', fontsize=16)

        # Return all artists that need to be updated
        return []

    # Create the animation
    anim = FuncAnimation(fig, run_generations, frames=max_frames,
                         interval=200, blit=False)

    plt.close(fig)  # Prevent display of the figure

    return HTML(anim.to_jshtml())


def visualize_binary_ga_solution(chromosome: List[int],
                                 title: str = "Binary Solution",
                                 figsize: Tuple[int, int] = (10, 4)):
    """
    Visualize a binary chromosome from a genetic algorithm.

    Args:
        chromosome: Binary chromosome (list of 0s and 1s)
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)

    # Create a grid for the bits
    n = len(chromosome)
    rows = 1
    cols = n

    # If the chromosome is too long, arrange in multiple rows
    if n > 50:
        rows = math.ceil(n / 50)
        cols = min(n, 50)

    # Create the plot
    plt.imshow([chromosome], cmap='binary', aspect='auto')

    # Add gridlines
    plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5)

    # Add bit indices
    if n <= 50:
        plt.xticks(range(n), [str(i) for i in range(n)])
    else:
        plt.xticks([])

    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.show()


def animate_binary_ga(ga_instance, fitness_function,
                      iterations_per_frame: int = 1,
                      max_frames: int = 100,
                      figsize: Tuple[int, int] = (12, 6)):
    """
    Create an animation for a binary genetic algorithm.

    Args:
        ga_instance: Instance of the BinaryGeneticAlgorithm class
        fitness_function: Function to evaluate fitness
        iterations_per_frame: Number of generations to run between frames
        max_frames: Maximum number of frames in the animation
        figsize: Figure size as (width, height) tuple

    Returns:
        Animation HTML that can be displayed in a notebook
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Store history
    fitness_history = []
    best_fitness_history = []
    best_chromosomes = []

    # Initialize population if needed
    if not hasattr(ga_instance, 'population') or ga_instance.population is None:
        population = ga_instance.initialize_population()
    else:
        population = ga_instance.population

    # Set up the run function that will be called to make each frame
    def run_generations(frame):
        nonlocal ga_instance, population

        # Run the algorithm for a few iterations
        for _ in range(iterations_per_frame):
            # Evaluate population
            evaluated_population = [(chrom, fitness_function(chrom)) for chrom in population]

            # Get best and average fitness
            current_best = max(evaluated_population, key=lambda x: x[1])
            avg_fitness = sum(fitness for _, fitness in evaluated_population) / len(evaluated_population)

            # Store history
            fitness_history.append(avg_fitness)
            best_fitness_history.append(current_best[1])
            best_chromosomes.append(current_best[0])

            # Create next generation
            parents = ga_instance.select_parents(evaluated_population)
            offspring = ga_instance.crossover(parents)
            population = ga_instance.mutate(offspring)

        # Clear the axes for new frame
        ax1.clear()
        ax2.clear()

        # Plot fitness history
        generations = range(1, len(fitness_history) + 1)
        ax1.plot(generations, fitness_history, 'b-', label='Average Fitness')
        ax1.plot(generations, best_fitness_history, 'r-', label='Best Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Visualize best chromosome as a binary grid
        best_chromosome = best_chromosomes[-1]
        ax2.imshow([best_chromosome], cmap='binary', aspect='auto')
        ax2.set_title(f'Best Solution (Fitness: {best_fitness_history[-1]:.4f})')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Add generation number
        fig.suptitle(f'Generation {len(fitness_history)}', fontsize=16)

        # Return all artists that need to be updated
        return []

    # Create the animation
    anim = FuncAnimation(fig, run_generations, frames=max_frames,
                         interval=200, blit=False)

    plt.close(fig)  # Prevent display of the figure

    return HTML(anim.to_jshtml())


def visualize_differential_evolution(de_instance, fitness_function, bounds,
                                     title: str = "Differential Evolution",
                                     figsize: Tuple[int, int] = (12, 6)):
    """
    Visualize the optimization process of Differential Evolution for 2D problems.

    Args:
        de_instance: Instance of the DifferentialEvolution class
        fitness_function: Function to minimize
        bounds: List of (min, max) bounds for each variable
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    # Check if the problem is 2D
    if len(bounds) != 2:
        print("This visualization only works for 2D problems (exactly 2 variables)")
        return

    # Create meshgrid for contour plot
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate function values
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = fitness_function([X[j, i], Y[j, i]])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Run the DE algorithm
    population = de_instance.initialize_population()
    fitness_values = de_instance.evaluate_population(population)

    best_fitness_history = []
    avg_fitness_history = []

    for generation in range(de_instance.max_generations):
        # Create mutant vectors
        mutants = de_instance.mutation(population, fitness_values)

        # Create trial vectors
        trials = de_instance.crossover(population, mutants)

        # Selection
        population, fitness_values = de_instance.selection(population, trials, fitness_values)

        # Record statistics
        best_idx = fitness_values.index(min(fitness_values))
        best_fitness_history.append(fitness_values[best_idx])
        avg_fitness_history.append(sum(fitness_values) / len(fitness_values))

        # Plot every 10 generations or at the end
        if generation % 10 == 0 or generation == de_instance.max_generations - 1:
            # Clear previous plots
            ax1.clear()
            ax2.clear()

            # Plot contour and population
            contour = ax1.contourf(X, Y, Z, 50, cmap='viridis')
            fig.colorbar(contour, ax=ax1)

            # Extract x and y coordinates of individuals
            x_vals = [ind[0] for ind in population]
            y_vals = [ind[1] for ind in population]

            # Plot population
            ax1.scatter(x_vals, y_vals, color='red', marker='o')

            # Highlight best individual
            best_ind = population[best_idx]
            ax1.scatter(best_ind[0], best_ind[1], color='white', marker='*', s=200,
                        edgecolor='black', zorder=5)

            ax1.set_title(f'Population at Generation {generation}')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')

            # Plot fitness history
            generations = range(1, len(best_fitness_history) + 1)
            ax2.plot(generations, best_fitness_history, 'r-', label='Best Fitness')
            ax2.plot(generations, avg_fitness_history, 'b-', label='Average Fitness')

            ax2.set_title('Fitness Evolution')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Fitness')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)

            fig.suptitle(f'{title} - Generation {generation}', fontsize=16)
            plt.tight_layout()
            plt.pause(0.1)

    plt.show()


def animate_differential_evolution(de_instance, fitness_function, bounds,
                                   iterations_per_frame: int = 1,
                                   max_frames: int = 100,
                                   figsize: Tuple[int, int] = (12, 6)):
    """
    Create an animation for Differential Evolution optimization.

    Args:
        de_instance: Instance of the DifferentialEvolution class
        fitness_function: Function to minimize
        bounds: List of (min, max) bounds for each variable
        iterations_per_frame: Number of generations to run between frames
        max_frames: Maximum number of frames in the animation
        figsize: Figure size as (width, height) tuple

    Returns:
        Animation HTML that can be displayed in a notebook
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Store history
    best_fitness_history = []
    avg_fitness_history = []
    population_history = []

    # Initialize population
    population = de_instance.initialize_population()
    fitness_values = de_instance.evaluate_population(population)

    # If 2D problem, create contour plot
    if len(bounds) == 2:
        # Create meshgrid for contour plot
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)

        # Calculate function values
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = fitness_function([X[j, i], Y[j, i]])

    # Set up the run function that will be called to make each frame
    def run_generations(frame):
        nonlocal de_instance, population, fitness_values

        # Run the algorithm for a few iterations
        for _ in range(iterations_per_frame):
            # Create mutant vectors
            mutants = de_instance.mutation(population, fitness_values)

            # Create trial vectors
            trials = de_instance.crossover(population, mutants)

            # Selection
            population, fitness_values = de_instance.selection(population, trials, fitness_values)

            # Record statistics
            best_idx = fitness_values.index(min(fitness_values))
            best_fitness_history.append(fitness_values[best_idx])
            avg_fitness_history.append(sum(fitness_values) / len(fitness_values))
            population_history.append(population.copy())

        # Clear the axes for new frame
        ax1.clear()
        ax2.clear()

        # If 2D problem, plot contour and population
        if len(bounds) == 2:
            contour = ax1.contourf(X, Y, Z, 50, cmap='viridis')

            # Extract x and y coordinates of individuals
            x_vals = [ind[0] for ind in population]
            y_vals = [ind[1] for ind in population]

            # Plot population
            ax1.scatter(x_vals, y_vals, color='red', marker='o')

            # Highlight best individual
            best_idx = fitness_values.index(min(fitness_values))
            best_ind = population[best_idx]
            ax1.scatter(best_ind[0], best_ind[1], color='white', marker='*', s=200,
                        edgecolor='black', zorder=5)

            ax1.set_title('Current Population')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
        else:
            # For higher dimensions, show best solution values
            best_idx = fitness_values.index(min(fitness_values))
            best_ind = population[best_idx]

            ax1.axis('off')
            ax1.text(0.5, 0.5, f"Best Solution:\n{best_ind}\n\nFitness: {fitness_values[best_idx]:.6f}",
                     ha='center', va='center', fontsize=12)
            ax1.set_title('Current Best Solution')

        # Plot fitness history
        generations = range(1, len(best_fitness_history) + 1)
        ax2.plot(generations, best_fitness_history, 'r-', label='Best Fitness')
        ax2.plot(generations, avg_fitness_history, 'b-', label='Average Fitness')

        ax2.set_title('Fitness Evolution')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Add generation number
        fig.suptitle(f'Generation {len(best_fitness_history)}', fontsize=16)

        # Return all artists that need to be updated
        return []

    # Create the animation
    anim = FuncAnimation(fig, run_generations, frames=max_frames,
                         interval=200, blit=False)

    plt.tight_layout()
    plt.close(fig)  # Prevent display of the figure

    return HTML(anim.to_jshtml())