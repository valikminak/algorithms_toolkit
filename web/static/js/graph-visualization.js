// graph-visualization.js - Specialized visualization for graph algorithms

/**
 * A specialized visualization system for graph algorithms using Canvas
 */
class GraphVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.graph = null;
        this.frames = [];
        this.currentFrame = 0;
        this.layout = {};
        this.nodeRadius = 20;
        this.animationSpeed = 1;
        this.animationTimer = null;
        this.paused = true;
        this.isInitialized = false;

        // Visual properties
        this.colors = {
            node: {
                default: '#4a6ee0',
                visited: '#28a745',
                current: '#e04a6e',
                queued: '#ffc107',
                start: '#9c27b0',
                end: '#ff5722',
                path: '#ff9800'
            },
            edge: {
                default: '#666666',
                path: '#ff9800',
                visited: '#28a745',
                considering: '#ffc107'
            },
            text: '#ffffff',
            background: '#f8f9fa'
        };

        // Bind methods
        this.render = this.render.bind(this);
        this.animate = this.animate.bind(this);
        this.handleResize = this.handleResize.bind(this);

        // Set up event listeners
        window.addEventListener('resize', this.handleResize);
    }

    /**
     * Initialize the visualizer with graph data and algorithm frames
     * @param {Object} data - The graph and visualization data
     */
    initialize(data) {
        // Clear any existing state
        this.reset();

        // Set up graph data
        this.graph = data.graph || {};
        this.frames = data.visualization || [];
        this.currentFrame = 0;

        // Generate layout if not provided
        if (!this.layout || Object.keys(this.layout).length === 0) {
            this.generateLayout();
        }

        this.isInitialized = true;

        // Render the first frame
        this.render();
    }

    /**
     * Reset the visualizer state
     */
    reset() {
        if (this.animationTimer) {
            clearInterval(this.animationTimer);
            this.animationTimer = null;
        }

        this.graph = null;
        this.frames = [];
        this.currentFrame = 0;
        this.layout = {};
        this.paused = true;
        this.isInitialized = false;

        // Clear canvas
        this.clearCanvas();
    }

    /**
     * Clear the canvas
     */
    clearCanvas() {
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    /**
     * Generate a force-directed layout for the graph
     */
    generateLayout() {
        if (!this.graph || !this.graph.vertices) return;

        const vertices = this.graph.vertices || [];
        const edges = this.graph.edges || {};

        // Initialize random positions
        const layout = {};
        const padding = this.nodeRadius * 3;
        const width = this.canvas.width - (padding * 2);
        const height = this.canvas.height - (padding * 2);

        // Generate initial positions in a circle
        const centerX = width / 2 + padding;
        const centerY = height / 2 + padding;
        const radius = Math.min(width, height) / 2.5;

        vertices.forEach((vertex, i) => {
            const angle = (i / vertices.length) * 2 * Math.PI;
            layout[vertex] = {
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle)
            };
        });

        // Apply simple force-directed algorithm to improve layout
        const simulateForces = () => {
            // Repulsive forces between all nodes
            for (let i = 0; i < vertices.length; i++) {
                for (let j = i + 1; j < vertices.length; j++) {
                    const v1 = vertices[i];
                    const v2 = vertices[j];

                    const dx = layout[v2].x - layout[v1].x;
                    const dy = layout[v2].y - layout[v1].y;
                    const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                    // Repulsive force (inversely proportional to distance)
                    const repulsionForce = 1000 / distance;
                    const fx = dx / distance * repulsionForce;
                    const fy = dy / distance * repulsionForce;

                    // Apply forces (move away from each other)
                    layout[v1].x -= fx;
                    layout[v1].y -= fy;
                    layout[v2].x += fx;
                    layout[v2].y += fy;
                }
            }

            // Attractive forces along edges
            for (const [edgeKey, weight] of Object.entries(edges)) {
                const [v1, v2] = edgeKey.split(',');

                const dx = layout[v2].x - layout[v1].x;
                const dy = layout[v2].y - layout[v1].y;
                const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                // Attractive force (proportional to distance)
                const attractionForce = distance * 0.05;
                const fx = dx / distance * attractionForce;
                const fy = dy / distance * attractionForce;

                // Apply forces (move toward each other)
                layout[v1].x += fx;
                layout[v1].y += fy;
                layout[v2].x -= fx;
                layout[v2].y -= fy;
            }

            // Keep nodes within bounds
            for (const vertex of vertices) {
                layout[vertex].x = Math.max(padding, Math.min(width + padding, layout[vertex].x));
                layout[vertex].y = Math.max(padding, Math.min(height + padding, layout[vertex].y));
            }
        };

        // Run force simulation for a few iterations
        for (let i = 0; i < 100; i++) {
            simulateForces();
        }

        this.layout = layout;
    }

    /**
     * Render the current frame of the visualization
     */
    render() {
        if (!this.isInitialized || this.frames.length === 0) return;

        const frame = this.frames[this.currentFrame];

        // Clear canvas
        this.clearCanvas();

        // Draw background
        this.ctx.fillStyle = this.colors.background;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Get current frame data
        const visitedVertices = frame.visited || [];
        const currentVertex = frame.current;
        const queuedVertices = frame.queue || [];
        const path = frame.path || [];
        const consideredEdges = frame.consideredEdges || [];
        const pathEdges = frame.pathEdges || [];
        const startVertex = frame.start;
        const endVertex = frame.end;

        // Draw edges first (so they appear under nodes)
        this.drawEdges(pathEdges, consideredEdges);

        // Draw nodes
        this.drawNodes(
            visitedVertices,
            currentVertex,
            queuedVertices,
            path,
            startVertex,
            endVertex
        );

        // Draw distances if available (for Dijkstra/A*)
        if (frame.distances) {
            this.drawDistances(frame.distances);
        }

        // Draw frame info
        this.drawFrameInfo(frame.info || '', this.currentFrame + 1, this.frames.length);
    }

    /**
     * Draw all edges in the graph
     * @param {Array} pathEdges - Edges that are part of the current path
     * @param {Array} consideredEdges - Edges being considered in this step
     */
    drawEdges(pathEdges = [], consideredEdges = []) {
        if (!this.graph || !this.graph.edges) return;

        const edges = this.graph.edges || {};

        for (const edgeKey in edges) {
            const [v1, v2] = edgeKey.split(',');
            const weight = edges[edgeKey];

            // Determine edge color based on state
            let color = this.colors.edge.default;
            let lineWidth = 2;

            if (pathEdges.includes(edgeKey) || pathEdges.includes(`${v2},${v1}`)) {
                color = this.colors.edge.path;
                lineWidth = 3;
            } else if (consideredEdges.includes(edgeKey) || consideredEdges.includes(`${v2},${v1}`)) {
                color = this.colors.edge.considering;
                lineWidth = 2.5;
            }

            // Draw the edge
            const p1 = this.layout[v1];
            const p2 = this.layout[v2];

            if (p1 && p2) {
                this.ctx.beginPath();
                this.ctx.moveTo(p1.x, p1.y);
                this.ctx.lineTo(p2.x, p2.y);
                this.ctx.strokeStyle = color;
                this.ctx.lineWidth = lineWidth;
                this.ctx.stroke();

                // Draw weight if the graph is weighted
                if (this.graph.weighted) {
                    const midX = (p1.x + p2.x) / 2;
                    const midY = (p1.y + p2.y) / 2;

                    // Draw weight background
                    this.ctx.fillStyle = 'white';
                    this.ctx.beginPath();
                    this.ctx.arc(midX, midY, 12, 0, 2 * Math.PI);
                    this.ctx.fill();

                    // Draw weight text
                    this.ctx.fillStyle = '#333';
                    this.ctx.font = '12px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText(weight, midX, midY);
                }
            }
        }
    }

    /**
     * Draw all nodes in the graph
     * @param {Array} visitedVertices - Vertices that have been visited
     * @param {string} currentVertex - The current vertex being explored
     * @param {Array} queuedVertices - Vertices in the queue/frontier
     * @param {Array} path - Vertices in the current path
     * @param {string} startVertex - The starting vertex
     * @param {string} endVertex - The target/end vertex
     */
    drawNodes(
        visitedVertices = [],
        currentVertex = null,
        queuedVertices = [],
        path = [],
        startVertex = null,
        endVertex = null
    ) {
        if (!this.graph || !this.graph.vertices) return;

        const vertices = this.graph.vertices || [];

        for (const vertex of vertices) {
            const position = this.layout[vertex];
            if (!position) continue;

            // Determine node color based on state
            let color = this.colors.node.default;

            // Priority of color selection: current > path > queued > visited > start/end > default
            if (vertex === currentVertex) {
                color = this.colors.node.current;
            } else if (path.includes(vertex)) {
                color = this.colors.node.path;
            } else if (queuedVertices.includes(vertex)) {
                color = this.colors.node.queued;
            } else if (visitedVertices.includes(vertex)) {
                color = this.colors.node.visited;
            } else if (vertex === startVertex) {
                color = this.colors.node.start;
            } else if (vertex === endVertex) {
                color = this.colors.node.end;
            }

            // Draw node
            this.ctx.beginPath();
            this.ctx.arc(position.x, position.y, this.nodeRadius, 0, 2 * Math.PI);
            this.ctx.fillStyle = color;
            this.ctx.fill();
            this.ctx.strokeStyle = '#333';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            // Draw node label
            this.ctx.fillStyle = this.colors.text;
            this.ctx.font = '14px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(vertex, position.x, position.y);
        }
    }

    /**
     * Draw distance labels for Dijkstra or A* algorithm
     * @param {Object} distances - Map of vertices to their current distances
     */
    drawDistances(distances) {
        for (const [vertex, distance] of Object.entries(distances)) {
            const position = this.layout[vertex];
            if (!position) continue;

            // Don't display infinity directly
            const displayDistance = distance === Infinity ? '∞' : distance;

            // Draw distance background
            this.ctx.fillStyle = 'white';
            this.ctx.beginPath();
            this.ctx.arc(position.x, position.y + this.nodeRadius + 10, 12, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.strokeStyle = '#333';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();

            // Draw distance text
            this.ctx.fillStyle = '#333';
            this.ctx.font = '11px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(displayDistance, position.x, position.y + this.nodeRadius + 10);
        }
    }

    /**
     * Draw frame information text
     * @param {string} info - Information about the current step
     * @param {number} currentFrame - Current frame number
     * @param {number} totalFrames - Total number of frames
     */
    drawFrameInfo(info, currentFrame, totalFrames) {
        // Draw frame progress
        this.ctx.fillStyle = '#333';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'top';
        this.ctx.fillText(`Step ${currentFrame} of ${totalFrames}`, 10, 10);

        // Draw algorithm info
        this.ctx.fillStyle = '#333';
        this.ctx.font = '14px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';

        // Handle multi-line text
        const maxWidth = this.canvas.width - 20;
        const lineHeight = 20;

        if (info.length > 80) {
            // Split long text into multiple lines
            const words = info.split(' ');
            let line = '';
            let y = this.canvas.height - 60;

            for (const word of words) {
                const testLine = line + word + ' ';
                const metrics = this.ctx.measureText(testLine);
                const testWidth = metrics.width;

                if (testWidth > maxWidth && line.length > 0) {
                    this.ctx.fillText(line, this.canvas.width / 2, y);
                    line = word + ' ';
                    y += lineHeight;
                } else {
                    line = testLine;
                }
            }

            this.ctx.fillText(line, this.canvas.width / 2, y);
        } else {
            // Single line text
            this.ctx.fillText(info, this.canvas.width / 2, this.canvas.height - 40);
        }
    }

    /**
     * Start the animation
     * @param {number} speed - Animation speed (1-10)
     */
    startAnimation(speed) {
        if (!this.isInitialized) return;

        this.animationSpeed = speed || 1;
        this.paused = false;

        // Clear any existing timer
        if (this.animationTimer) {
            clearInterval(this.animationTimer);
        }

        this.animate();
    }

    /**
     * Animate the visualization
     */
    animate() {
        if (this.paused) return;

        // Calculate delay based on speed (1-10)
        const delay = 1000 / this.animationSpeed;

        this.animationTimer = setInterval(() => {
            this.render();

            // Advance to next frame
            if (this.currentFrame < this.frames.length - 1) {
                this.currentFrame++;
            } else {
                // Stop at the end
                this.pauseAnimation();
            }
        }, delay);
    }

    /**
     * Pause the animation
     */
    pauseAnimation() {
        this.paused = true;

        if (this.animationTimer) {
            clearInterval(this.animationTimer);
            this.animationTimer = null;
        }
    }

    /**
     * Go to a specific frame
     * @param {number} frameIndex - Index of the frame to show
     */
    goToFrame(frameIndex) {
        if (!this.isInitialized) return;

        // Pause any running animation
        this.pauseAnimation();

        // Validate frame index
        this.currentFrame = Math.max(0, Math.min(frameIndex, this.frames.length - 1));

        // Render the frame
        this.render();
    }

    /**
     * Go to the next frame
     */
    nextFrame() {
        if (!this.isInitialized) return;

        // Pause any running animation
        this.pauseAnimation();

        // Advance to next frame if possible
        if (this.currentFrame < this.frames.length - 1) {
            this.currentFrame++;
            this.render();
        }
    }

    /**
     * Go to the previous frame
     */
    prevFrame() {
        if (!this.isInitialized) return;

        // Pause any running animation
        this.pauseAnimation();

        // Go to previous frame if possible
        if (this.currentFrame > 0) {
            this.currentFrame--;
            this.render();
        }
    }

    /**
     * Set the animation speed
     * @param {number} speed - Speed value (1-10)
     */
    setSpeed(speed) {
        this.animationSpeed = speed;

        // Restart animation if it's running
        if (!this.paused && this.animationTimer) {
            this.pauseAnimation();
            this.startAnimation(speed);
        }
    }

    /**
     * Handle window resize to make the canvas responsive
     */
    handleResize() {
        if (!this.canvas || !this.canvas.parentElement) return;

        // Save current frame and state
        const currentFrame = this.currentFrame;
        const wasPaused = this.paused;

        // Pause animation
        this.pauseAnimation();

        // Resize canvas to fit container
        this.canvas.width = this.canvas.parentElement.clientWidth || 800;
        this.canvas.height = 400; // Fixed height or you can make this dynamic too

        // Recalculate layout
        this.generateLayout();

        // Restore state
        this.currentFrame = currentFrame;

        // Render with new dimensions
        this.render();

        // Resume animation if it was running
        if (!wasPaused) {
            this.startAnimation(this.animationSpeed);
        }
    }

    /**
     * Attach the visualizer to animation controls
     * @param {Object} controls - Control button elements
     */
    attachControls(controls) {
        if (!controls) return;

        // Play/Pause button
        if (controls.playPause) {
            controls.playPause.addEventListener('click', () => {
                if (this.paused) {
                    this.startAnimation(this.animationSpeed);
                    controls.playPause.textContent = '⏸️';
                } else {
                    this.pauseAnimation();
                    controls.playPause.textContent = '▶️';
                }
            });
        }

        // Next frame button
        if (controls.next) {
            controls.next.addEventListener('click', () => {
                this.nextFrame();
                if (controls.playPause) {
                    controls.playPause.textContent = '▶️';
                }
            });
        }

        // Previous frame button
        if (controls.prev) {
            controls.prev.addEventListener('click', () => {
                this.prevFrame();
                if (controls.playPause) {
                    controls.playPause.textContent = '▶️';
                }
            });
        }

        // Restart button
        if (controls.restart) {
            controls.restart.addEventListener('click', () => {
                this.goToFrame(0);
                if (controls.playPause) {
                    controls.playPause.textContent = '▶️';
                }
            });
        }

        // Progress bar
        if (controls.progressBar) {
            controls.progressBar.addEventListener('click', (e) => {
                const rect = controls.progressBar.getBoundingClientRect();
                const clickPosition = e.clientX - rect.left;
                const percentage = clickPosition / rect.width;

                // Calculate frame index based on percentage
                const frameIndex = Math.floor(percentage * this.frames.length);
                this.goToFrame(frameIndex);

                // Update progress indicator if available
                if (controls.progressIndicator) {
                    controls.progressIndicator.style.width = `${percentage * 100}%`;
                }

                if (controls.playPause) {
                    controls.playPause.textContent = '▶️';
                }
            });
        }
    }
}

// Export the visualizer
export default GraphVisualizer;