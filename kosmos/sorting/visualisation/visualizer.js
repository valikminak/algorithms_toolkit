/**
 * Specialized visualizer for sorting algorithms.
 * Extends the base visualizer with sorting-specific rendering.
 */
class SortingVisualizer extends VisualizerBase {
    /**
     * Create a sorting visualizer
     * @param {string} containerId - ID of the HTML container element
     * @param {Object} options - Visualization options
     */
    constructor(containerId, options = {}) {
        // Set default options for sorting visualization
        const sortingOptions = {
            barSpacing: options.barSpacing || 0.2,
            maxBarSpacing: options.maxBarSpacing || 4,
            labelHeight: options.labelHeight || 40,
            colors: {
                ...options.colors,
                bar: options.colors?.bar || '#4a6ee0',
                highlight: options.colors?.highlight || '#e04a6e',
                sorted: options.colors?.sorted || '#28a745'
            },
            ...options
        };

        super(containerId, sortingOptions);
    }

    /**
     * Render a sorting algorithm frame
     */
    render() {
        if (!this.frames.length || this.currentFrame >= this.frames.length) {
            this.showPlaceholder("No sorting data to visualize");
            return;
        }

        const frame = this.frames[this.currentFrame];

        // Update info text
        if (frame.info) {
            this.updateInfoText(frame.info);
        }

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Render array state
        this.renderArrayState(frame);
    }

    /**
     * Render the array state for a sorting algorithm frame
     * @param {Object} frame - Frame data containing array state
     */
    renderArrayState(frame) {
        if (!frame || !frame.state) {
            this.ctx.fillStyle = this.options.colors.background;
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.fillStyle = this.options.colors.lightText;
            this.ctx.font = `16px ${this.options.fontFamily}`;
            this.ctx.textAlign = 'center';
            this.ctx.fillText('No array data available', this.canvas.width / 2, this.canvas.height / 2);
            return;
        }

        const array = Array.isArray(frame.state) ? frame.state : [];

        // No data to display
        if (array.length === 0) {
            this.ctx.fillStyle = this.options.colors.background;
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.fillStyle = this.options.colors.lightText;
            this.ctx.font = `16px ${this.options.fontFamily}`;
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Empty array', this.canvas.width / 2, this.canvas.height / 2);
            return;
        }

        // Find max value safely
        const maxValue = Math.max(...array.map(v => isNaN(v) ? 0 : v), 1);

        const barWidth = this.canvas.width / array.length;
        const barSpacing = Math.min(barWidth * this.options.barSpacing, this.options.maxBarSpacing);

        // Draw bars
        for (let i = 0; i < array.length; i++) {
            const value = array[i];
            const barHeight = (value / maxValue) * (this.canvas.height - this.options.labelHeight - 40);

            // Determine bar color (highlight current elements being processed)
            let color = this.options.colors.bar;

            if (frame.sorted && frame.sorted.includes(i)) {
                color = this.options.colors.sorted; // Element is in final sorted position
            } else if (frame.highlight && frame.highlight.includes(i)) {
                color = this.options.colors.highlight; // Element is being compared/swapped
            }

            // Draw bar
            this.ctx.fillStyle = color;
            this.ctx.fillRect(
                i * barWidth + barSpacing / 2,
                this.canvas.height - barHeight - this.options.labelHeight,
                barWidth - barSpacing,
                barHeight
            );

            // Draw value label if there's enough space
            if (barWidth > 20) {
                this.ctx.fillStyle = this.options.colors.text;
                this.ctx.font = `${this.options.fontSize}px ${this.options.fontFamily}`;
                this.ctx.textAlign = 'center';
                this.ctx.fillText(
                    value.toString(),
                    i * barWidth + barWidth / 2,
                    this.canvas.height - barHeight - this.options.labelHeight - 5
                );
            }

            // Draw index label
            this.ctx.fillStyle = this.options.colors.lightText;
            this.ctx.font = `${this.options.fontSize - 2}px ${this.options.fontFamily}`;
            this.ctx.textAlign = 'center';
            this.ctx.fillText(
                i.toString(),
                i * barWidth + barWidth / 2,
                this.canvas.height - 20
            );
        }

        // Draw frame number
        this.ctx.fillStyle = this.options.colors.text;
        this.ctx.font = `${this.options.fontSize}px ${this.options.fontFamily}`;
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`Frame: ${this.currentFrame + 1}/${this.frames.length}`, 10, 20);

        // Draw additional info if available
        if (frame.comparisonCount !== undefined) {
            this.ctx.fillText(`Comparisons: ${frame.comparisonCount}`, 10, 40);
        }

        if (frame.swapCount !== undefined) {
            this.ctx.fillText(`Swaps: ${frame.swapCount}`, 10, 60);
        }
    }

    /**
     * Convert algorithm result data to visualization frames
     * @param {Object} algorithmData - Data returned from the algorithm API
     * @returns {Array} - Array of frames for visualization
     */
    static prepareVisualizationData(algorithmData) {
        if (!algorithmData || !algorithmData.visualization || !Array.isArray(algorithmData.visualization)) {
            return [];
        }

        return algorithmData.visualization.map(frame => {
            // Ensure frame has all required properties
            return {
                state: frame.state || [],
                highlight: frame.highlight || [],
                sorted: frame.sorted || [],
                info: frame.info || '',
                comparisonCount: frame.comparisonCount,
                swapCount: frame.swapCount
            };
        });
    }
}

// Export the class if in a module environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SortingVisualizer;
} else if (typeof define === 'function' && define.amd) {
    define(['./visualizer_base'], function(VisualizerBase) {
        window.VisualizerBase = VisualizerBase;
        return SortingVisualizer;
    });
} else {
    window.SortingVisualizer = SortingVisualizer;
}