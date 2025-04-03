import io
import base64
import matplotlib.pyplot as plt


def convert_plot_to_image(plot, format='png', dpi=100):
    """Convert a matplotlib figure to a base64 encoded image"""
    buf = io.BytesIO()
    plot.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/{format};base64,{img_str}'


def convert_animation_to_frames(animation, num_frames=30):
    """
    Convert a matplotlib animation to a list of frames
    suitable for web playback
    """
    frames = []

    # Extract data from the animation
    fig = animation._fig
    animation._drawn_artists

    # This is a simplification - in reality, it's more complex to extract
    # the exact state information from a matplotlib animation

    # For sorting visualizations, we'll create a sequence of snapshots
    # This is a placeholder implementation
    for i in range(num_frames):
        # Capture the figure at this frame
        animation._draw_frame(i % animation.save_count)

        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        frames.append({
            'image': f'data:image/png;base64,{img_str}',
            'frame': i
        })

    return frames


def extract_array_from_bar_chart(ax):
    """Extract array values from a bar chart for web visualization"""
    heights = [bar.get_height() for bar in ax.patches]
    return heights


def extract_graph_data_from_networkx(ax):
    """Extract node and edge positions from a networkx graph visualization"""
    # This is a placeholder - actual implementation would
    # extract positions, colors, and other attributes
    node_collection = next((child for child in ax.get_children()
                            if isinstance(child, plt.matplotlib.collections.PathCollection)), None)

    if node_collection:
        positions = node_collection.get_offsets().data
        # Convert to list of [x, y] pairs
        nodes = positions.tolist()
    else:
        nodes = []

    # Extract edges (lines)
    line_collection = next((child for child in ax.get_children()
                            if isinstance(child, plt.matplotlib.collections.LineCollection)), None)

    if line_collection:
        segments = line_collection.get_segments()
        # Convert to list of [[x1, y1], [x2, y2]] pairs
        edges = segments.tolist()
    else:
        edges = []

    return {
        'nodes': nodes,
        'edges': edges
    }