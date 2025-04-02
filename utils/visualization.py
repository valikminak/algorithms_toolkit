from typing import Any, Callable, List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


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


def visualize_sorting_algorithm(algorithm: Callable, arr: List[int], title: str = None):
    """
    Visualize a sorting algorithm using an animation.

    Args:
        algorithm: The sorting algorithm function
        arr: The array to sort
        title: The title of the animation

    Returns:
        The animation object
    """
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
    ax.set_ylim(0, max(history[0]) * 1.1 if history[0] else 10)

    # Update function for animation
    def update(frame):
        for i, val in enumerate(history[frame]):
            bars[i].set_height(val)
        return bars

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(history), interval=200, repeat=False)

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