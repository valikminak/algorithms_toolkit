import io
import base64


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

    # For each frame in the animation
    for i in range(min(num_frames, animation.save_count)):
        # Capture the figure at this frame
        animation._draw_frame(i)

        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        frames.append({
            'image': f'data:image/png;base64,{img_str}',
            'frame': i,
            'info': f'Frame {i}'  # Simple info text
        })

    return frames


def extract_array_data(ax):
    """Extract array data from a visualization for web rendering"""
    # Extract data from bar charts
    if ax.patches and len(ax.patches) > 0:
        # This is likely a bar chart
        heights = [patch.get_height() for patch in ax.patches]
        return heights

    # Extract data from line plots
    elif ax.lines and len(ax.lines) > 0:
        # This is likely a line plot
        line_data = []
        for line in ax.lines:
            x_data = line.get_xdata().tolist()
            y_data = line.get_ydata().tolist()
            line_data.append({'x': x_data, 'y': y_data})
        return line_data

    return None


def convert_visualization_to_json(vis_obj, algorithm_type):
    """
    Convert a visualization object to JSON format for web rendering

    Args:
        vis_obj: Visualization object from utils.visualization
        algorithm_type: Type of algorithm ('sorting', 'searching', etc.)

    Returns:
        JSON-serializable data for web visualization
    """
    result = {
        'type': algorithm_type,
        'frames': []
    }

    # Handle different types of visualizations
    if hasattr(vis_obj, 'to_jshtml'):
        # This is an animation - convert to series of frames
        frames = convert_animation_to_frames(vis_obj)
        result['frames'] = frames
    elif hasattr(vis_obj, 'savefig'):
        # This is a static figure - convert to image
        result['frames'] = [{
            'image': convert_plot_to_image(vis_obj),
            'info': 'Static visualization'
        }]

    return result