// network-flow-visualization.js - Specialized visualization for network flow algorithms

/**
 * A visualization system for network flow algorithms
 */
class NetworkFlowVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.frames = [];
        this.currentFrame = 0;
        this.animationTimer = null;
        this.paused = true;
        this.graphLayout = {};

        // Visualization properties
        this.colors = {
            background: '#f8f9fa',
            node: {
                default: '#4a6ee0',
                source: '#28a745',
                sink: '#e04a6e',
                path: '#ff9800'
            },
            edge: {
                default: '#666666',
                path: '#ff9800',
                residual: '#aaaaaa',
                cut: '#e04a6e'
            },
            text: '#ffffff'
        };

        this.nodeRadius = 25;
        this.edgeWidth = 2;
        this.padding = 40;
        this.fontFamily = 'Arial, sans-serif';

        // Bind methods
        this.render = this.render.bind(this);
        this.animate = this.animate.bind(this);
        this.handleResize = this.handleResize.bind(this);

        // Set up event listeners
        window.addEventListener('resize', this.handleResize);
    }

    /**
     * Initialize the visualizer with network flow data
     * @param {Object} data - The visualization data
     */
    initialize(data) {
        this.reset();

        this.algorithm = data.algorithm || '';
        this.frames = data.steps || [];
        this.currentFrame = 0;

        // Extract flow network data
        this.graph = this.extractGraph(data);
        this.source = data.source || '';
        this.sink = data.sink || '';

        // Generate layout for the graph
        this.generateLayout();

        this.render();
    }

    /**
     * Extract graph data from visualization data
     */
    extractGraph(data) {
        let graph = {
            vertices: new Set(),
            edges: {}
        };

        // Extract vertices and edges from the first frame
        if (data.steps && data.steps.length > 0) {
            const firstFrame = data.steps[0];

            // Extract vertices from residual graph
            if (firstFrame.residual_graph) {
                for (const vertex in firstFrame.residual_graph) {
                    graph.vertices.add(vertex);

                    // Add neighbor vertices
                    const neighbors = firstFrame.residual_graph[vertex];
                    for (const neighbor in neighbors) {
                        graph.vertices.add(neighbor);
                    }
                }
            }

            // Extract edges from residual graph
            if (firstFrame.residual_graph) {
                for (const vertex in firstFrame.residual_graph) {
                    for (const neighbor in firstFrame.residual_graph[vertex]) {
                        const capacity = firstFrame.residual_graph[vertex][neighbor];
                        graph.edges[`${vertex},${neighbor}`] = {
                            capacity: capacity,
                            flow: 0
                        };
                    }
                }
            }

            // Update flow values if flow_graph is available
            if (firstFrame.flow_graph) {
                for (const vertex in firstFrame.flow_graph) {
                    for (const neighbor in firstFrame.flow_graph[vertex]) {
                        const flow = firstFrame.flow_graph[vertex][neighbor];
                        const edgeKey = `${vertex},${neighbor}`;

                        if (graph.edges[edgeKey]) {
                            graph.edges[edgeKey].flow = flow;
                        }
                    }
                }
            }
        }

        return graph;
    }

    /**
     * Reset the visualizer state
     */
    reset() {
        if (this.animationTimer) {
            clearInterval(this.animationTimer);
            this.animationTimer = null;
        }

        this.frames = [];
        this.currentFrame = 0;
        this.paused = true;

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
     * Generate a layout for the graph
     */
    generateLayout() {
        const vertices = Array.from(this.graph.vertices);

        // Assign source and sink positions
        const sourceIndex = vertices.indexOf(this.source);
        const sinkIndex = vertices.indexOf(this.sink);

        if (sourceIndex !== -1) {
            vertices.splice(sourceIndex, 1);
        }

        if (sinkIndex !== -1) {
            vertices.splice(sinkIndex < sourceIndex ? sinkIndex : sinkIndex - 1, 1);
        }

        // Calculate canvas dimensions and positions
        const width = this.canvas.width - this.padding * 2;
        const height = this.canvas.height - this.padding * 2;

        // Position source at left, sink at right
        this.graphLayout[this.source] = {
            x: this.padding + this.nodeRadius,
            y: height / 2 + this.padding
        };

        this.graphLayout[this.sink] = {
            x: width - this.nodeRadius + this.padding,
            y: height / 2 + this.padding
        };

        // Position other vertices in between
        const nodeCount = vertices.length;
        const midX = width / 2 + this.padding;

        if (nodeCount > 0) {
            // If there's an odd number of intermediate nodes, place one in the center
            if (nodeCount % 2 === 1) {
                const middle = Math.floor(nodeCount / 2);
                this.graphLayout[vertices[middle]] = {
                    x: midX,
                    y: height / 2 + this.padding
                };

                // Position nodes to the left of center
                for (let i = 0; i < middle; i++) {
                    this.graphLayout[vertices[i]] = {
                        x: this.padding + this.nodeRadius + (midX - this.padding - this.nodeRadius) * (i + 1) / (middle + 1),
                        y: this.padding + height / 3
                    };
                }

                // Position nodes to the right of center
                for (let i = middle + 1; i < nodeCount; i++) {
                    this.graphLayout[vertices[i]] = {
                        x: midX + (width - this.nodeRadius - midX) * (i - middle) / (nodeCount - middle),
                        y: this.padding + height * 2 / 3
                    };
                }
            }
            // If there's an even number, position half above and half below the center line
            else {
                const half = nodeCount / 2;

                // Position first half of nodes (top row)
                for (let i = 0; i < half; i++) {
                    this.graphLayout[vertices[i]] = {
                        x: this.padding + this.nodeRadius + (width - 2 * this.nodeRadius) * (i + 1) / (half + 1),
                        y: this.padding + height / 3
                    };
                }

                // Position second half of nodes (bottom row)
                for (let i = half; i < nodeCount; i++) {
                    this.graphLayout[vertices[i]] = {
                        x: this.padding + this.nodeRadius + (width - 2 * this.nodeRadius) * (i - half + 1) / (half + 1),
                        y: this.padding + height * 2 / 3
                    };
                }
            }
        }
    }

    /**
     * Render the current frame
     */
    render() {
        if (!this.frames.length) return;

        const frame = this.frames[this.currentFrame];
        if (!frame) return;

        this.clearCanvas();

        // Fill background
        this.ctx.fillStyle = this.colors.background;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Determine which rendering function to use based on algorithm
        if (this.algorithm === 'ford_fulkerson' || this.algorithm === 'edmonds_karp') {
            this.renderFlowNetworkFrame(frame);
        } else if (this.algorithm === 'min_cut') {
            this.renderMinCutFrame(frame);
        } else {
            this.renderGenericFlowFrame(frame);
        }

        // Draw frame info
        this.drawFrameInfo(frame.info || '', this.currentFrame + 1, this.frames.length);
    }

    /**
     * Render a flow network algorithm frame
     */
    renderFlowNetworkFrame(frame) {
        // Extract necessary data from the frame
        const residualGraph = frame.residual_graph || {};
        const flowGraph = frame.flow_graph || {};
        const path = frame.path || [];
        const maxFlowSoFar = frame.max_flow_so_far || 0;
        const pathFlow = frame.path_flow || 0;

        // Draw the residual graph edges first
        for (const edgeKey in residualGraph) {
            for (const neighbor in residualGraph[edgeKey]) {
                const capacity = residualGraph[edgeKey][neighbor];

                // Only draw if capacity > 0
                if (capacity > 0) {
                    const isInPath = path.length > 0 &&
                        path.indexOf(edgeKey) !== -1 &&
                        path[path.indexOf(edgeKey) + 1] === neighbor;

                    this.drawEdge(
                        edgeKey,
                        neighbor,
                        capacity,
                        0,
                        isInPath ? this.colors.edge.path : this.colors.edge.residual,
                        true
                    );
                }
            }
        }

        // Draw the flow graph edges
        for (const edgeKey in flowGraph) {
            for (const neighbor in flowGraph[edgeKey]) {
                const flow = flowGraph[edgeKey][neighbor];

                // Only draw if flow > 0
                if (flow > 0) {
                    const originalCapacity = this.getOriginalCapacity(edgeKey, neighbor);

                    this.drawEdge(
                        edgeKey,
                        neighbor,
                        originalCapacity,
                        flow,
                        this.colors.edge.default,
                        false
                    );
                }
            }
        }

        // Draw the vertices
        for (const vertex of this.graph.vertices) {
            const isInPath = path.includes(vertex);
            const isSource = vertex === this.source;
            const isSink = vertex === this.sink;

            let color = this.colors.node.default;
            if (isInPath) {
                color = this.colors.node.path;
            } else if (isSource) {
                color = this.colors.node.source;
            } else if (isSink) {
                color = this.colors.node.sink;
            }

            this.drawVertex(vertex, color);
        }

        // Draw max flow information
        this.ctx.font = '16px ' + this.fontFamily;
        this.ctx.fillStyle = '#333';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`Max Flow: ${maxFlowSoFar}`, this.padding, this.padding);

        if (path.length > 0) {
            this.ctx.fillText(`Augmenting Path: ${path.join(' → ')}`, this.padding, this.padding + 24);
            this.ctx.fillText(`Path Flow: ${pathFlow}`, this.padding, this.padding + 48);
        }
    }

    /**
     * Render a min-cut algorithm frame
     */
    renderMinCutFrame(frame) {
        // Extract data from the frame
        const minCut = frame.min_cut || 0;
        const cutEdges = frame.cut_edges || [];
        const sPartition = frame.s_partition || [];
        const tPartition = frame.t_partition || [];

        // Draw the graph edges
        for (const edgeKey in this.graph.edges) {
            const [u, v] = edgeKey.split(',');
            const edge = this.graph.edges[edgeKey];

            // Check if this edge is a cut edge
            const isCutEdge = cutEdges.some(e => e.from === u && e.to === v);

            this.drawEdge(
                u,
                v,
                edge.capacity,
                edge.flow,
                isCutEdge ? this.colors.edge.cut : this.colors.edge.default,
                false,
                isCutEdge ? 3 : 2  // Thicker lines for cut edges
            );
        }

        // Draw the vertices
        for (const vertex of this.graph.vertices) {
            const isSource = vertex === this.source;
            const isSink = vertex === this.sink;
            const isInSPartition = sPartition.includes(vertex);

            let color = this.colors.node.default;
            if (isSource) {
                color = this.colors.node.source;
            } else if (isSink) {
                color = this.colors.node.sink;
            } else if (isInSPartition) {
                color = this.colors.node.source;
            } else if (tPartition.includes(vertex)) {
                color = this.colors.node.sink;
            }

            this.drawVertex(vertex, color);
        }

        // Draw min-cut information
        this.ctx.font = '16px ' + this.fontFamily;
        this.ctx.fillStyle = '#333';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`Min Cut Value: ${minCut}`, this.padding, this.padding);

        if (cutEdges.length > 0) {
            const cutEdgesText = cutEdges.map(e => `(${e.from},${e.to})`).join(', ');
            this.ctx.fillText(`Cut Edges: ${cutEdgesText}`, this.padding, this.padding + 24);
        }

        if (sPartition.length > 0) {
            this.ctx.fillText(`S Partition: {${sPartition.join(', ')}}`, this.padding, this.padding + 48);
        }

        if (tPartition.length > 0) {
            this.ctx.fillText(`T Partition: {${tPartition.join(', ')}}`, this.padding, this.padding + 72);
        }
    }

    /**
     * Render a generic flow algorithm frame
     */
    renderGenericFlowFrame(frame) {
        // Extract residual graph data from frame if available
        const residualGraph = frame.residual_graph || {};

        // Draw edges
        for (const edgeKey in this.graph.edges) {
            const [u, v] = edgeKey.split(',');
            const edge = this.graph.edges[edgeKey];

            this.drawEdge(u, v, edge.capacity, edge.flow, this.colors.edge.default, false);
        }

        // Draw vertices
        for (const vertex of this.graph.vertices) {
            const isSource = vertex === this.source;
            const isSink = vertex === this.sink;

            let color = this.colors.node.default;
            if (isSource) {
                color = this.colors.node.source;
            } else if (isSink) {
                color = this.colors.node.sink;
            }

            this.drawVertex(vertex, color);
        }
    }

    /**
     * Draw a vertex (node) on the canvas
     */
    drawVertex(vertex, color) {
        const position = this.graphLayout[vertex];
        if (!position) return;

        // Draw circle
        this.ctx.beginPath();
        this.ctx.arc(position.x, position.y, this.nodeRadius, 0, 2 * Math.PI);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Draw vertex label
        this.ctx.fillStyle = this.colors.text;
        this.ctx.font = '14px ' + this.fontFamily;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(vertex, position.x, position.y);
    }

    /**
     * Draw an edge between two vertices
     */
    drawEdge(u, v, capacity, flow, color, isResidual, lineWidth = 2) {
        const posU = this.graphLayout[u];
        const posV = this.graphLayout[v];

        if (!posU || !posV) return;

        // Calculate direction vector
        const dx = posV.x - posU.x;
        const dy = posV.y - posU.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Normalize direction vector
        const nx = dx / distance;
        const ny = dy / distance;

        // Calculate the start and end points (adjusted for node radius)
        const startX = posU.x + nx * this.nodeRadius;
        const startY = posU.y + ny * this.nodeRadius;
        const endX = posV.x - nx * this.nodeRadius;
        const endY = posV.y - ny * this.nodeRadius;

        // Draw the edge line
        this.ctx.beginPath();
        this.ctx.moveTo(startX, startY);
        this.ctx.lineTo(endX, endY);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth;
        this.ctx.stroke();

        // Draw arrowhead
        const arrowSize = 8;
        const angle = Math.atan2(dy, dx);

        this.ctx.beginPath();
        this.ctx.moveTo(endX, endY);
        this.ctx.lineTo(
            endX - arrowSize * Math.cos(angle - Math.PI / 6),
            endY - arrowSize * Math.sin(angle - Math.PI / 6)
        );
        this.ctx.lineTo(
            endX - arrowSize * Math.cos(angle + Math.PI / 6),
            endY - arrowSize * Math.sin(angle + Math.PI / 6)
        );
        this.ctx.closePath();
        this.ctx.fillStyle = color;
        this.ctx.fill();

        // Draw capacity and flow labels
        const midX = (startX + endX) / 2;
        const midY = (startY + endY) / 2;

        // Offset the label slightly perpendicular to the edge
        const perpX = -ny * 15;  // Perpendicular to the edge
        const perpY = nx * 15;

        // Background for the label
        this.ctx.fillStyle = 'white';
        this.ctx.beginPath();
        this.ctx.ellipse(midX + perpX, midY + perpY, 24, 16, 0, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 1;
        this.ctx.stroke();

        // Draw label text
        this.ctx.fillStyle = '#333';
        this.ctx.font = '12px ' + this.fontFamily;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        if (isResidual) {
            // For residual network, just show the residual capacity
            this.ctx.fillText(capacity.toString(), midX + perpX, midY + perpY);
        } else {
            // For flow network, show flow/capacity
            this.ctx.fillText(`${flow}/${capacity}`, midX + perpX, midY + perpY);
        }
    }

    /**
     * Get the original capacity of an edge
     */
    getOriginalCapacity(u, v) {
        const edgeKey = `${u},${v}`;
        if (this.graph.edges[edgeKey]) {
            return this.graph.edges[edgeKey].capacity;
        }
        return 0;
    }

    /**
     * Draw frame information text
     */
    drawFrameInfo(info, currentFrame, totalFrames) {
        // Draw frame progress
        this.ctx.fillStyle = '#333';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'right';
        this.ctx.textBaseline = 'top';
        this.ctx.fillText(`Step ${currentFrame} of ${totalFrames}`, this.canvas.width - this.padding, this.padding);

        // Draw algorithm info
        this.ctx.fillStyle = '#333';
        this.ctx.font = '14px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'bottom';

        // Handle multi-line text
        if (info && info.length > 0) {
            const maxWidth = this.canvas.width - 2 * this.padding;
            const lineHeight = 20;
            const words = info.split(' ');
            let line = '';
            let y = this.canvas.height - this.padding;

            for (const word of words) {
                const testLine = line + (line ? ' ' : '') + word;
                const metrics = this.ctx.measureText(testLine);
                const testWidth = metrics.width;

                if (testWidth > maxWidth && line !== '') {
                    this.ctx.fillText(line, this.canvas.width / 2, y);
                    line = word;
                    y -= lineHeight;
                } else {
                    line = testLine;
                }
            }

            this.ctx.fillText(line, this.canvas.width / 2, y);
        }
    }

    /**
     * Start the animation
     */
    startAnimation(speed) {
        if (this.frames.length === 0) return;

        this.paused = false;

        if (this.animationTimer) {
            clearInterval(this.animationTimer);
        }

        const delay = 1000 / speed;

        this.animationTimer = setInterval(() => {
            if (this.paused) return;

            this.render();

            if (this.currentFrame < this.frames.length - 1) {
                this.currentFrame++;
            } else {
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
     */
    goToFrame(frameIndex) {
        if (this.frames.length === 0) return;

        this.pauseAnimation();
        this.currentFrame = Math.max(0, Math.min(frameIndex, this.frames.length - 1));
        this.render();
    }

    /**
     * Handle window resize
     */
    handleResize() {
        // Update canvas size
        const container = this.canvas.parentElement;
        if (container) {
            this.canvas.width = container.clientWidth;
            this.canvas.height = 500; // Fixed height for network flow visualization

            // Regenerate graph layout
            this.generateLayout();
            this.render();
        }
    }

    /**
     * Attach control buttons
     */
    attachControls(controls) {
        if (!controls) return;

        // Play/Pause button
        if (controls.playPause) {
            controls.playPause.addEventListener('click', () => {
                if (this.paused) {
                    this.startAnimation(5); // Default speed
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
                if (this.currentFrame < this.frames.length - 1) {
                    this.goToFrame(this.currentFrame + 1);
                }
                if (controls.playPause) {
                    controls.playPause.textContent = '▶️';
                }
            });
        }

        // Previous frame button
        if (controls.prev) {
            controls.prev.addEventListener('click', () => {
                if (this.currentFrame > 0) {
                    this.goToFrame(this.currentFrame - 1);
                }
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
                const percentage = (e.clientX - rect.left) / rect.width;
                const frameIndex = Math.floor(percentage * this.frames.length);

                this.goToFrame(frameIndex);

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

/**
 * Register the network flow visualizer with the main visualization system
 */
export function initNetworkFlowVisualization() {
    const visualizationContainer = document.getElementById('visualization-container');
    if (!visualizationContainer) return null;

    // Create canvas if it doesn't exist
    let canvas = document.getElementById('visualization-canvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'visualization-canvas';
        canvas.className = 'visualization-canvas';
        visualizationContainer.appendChild(canvas);
    }

    // Set canvas dimensions
    const width = visualizationContainer.clientWidth || 800;
    canvas.width = width;
    canvas.height = 500; // Fixed height for network flow visualization

    // Create network flow visualizer instance
    const flowVisualizer = new NetworkFlowVisualizer('visualization-canvas');

    // Create info text element if it doesn't exist
    let infoText = document.getElementById('info-text');
    if (!infoText) {
        infoText = document.createElement('div');
        infoText.id = 'info-text';
        infoText.className = 'info-text';
        visualizationContainer.appendChild(infoText);
    }

    // Create animation controls if they don't exist
    let controlsContainer = document.querySelector('.animation-controls');
    if (!controlsContainer) {
        controlsContainer = document.createElement('div');
        controlsContainer.className = 'animation-controls';

        controlsContainer.innerHTML = `
            <button class="restart" title="Restart">⏮️</button>
            <button class="step-backward" title="Previous Step">⏪</button>
            <button class="play-pause" title="Play/Pause">▶️</button>
            <button class="step-forward" title="Next Step">⏩</button>
            <div class="progress-bar">
                <div class="progress-bar-fill"></div>
            </div>
        `;

        visualizationContainer.appendChild(controlsContainer);
    }

    // Attach controls
    const controls = {
        playPause: document.querySelector('.play-pause'),
        next: document.querySelector('.step-forward'),
        prev: document.querySelector('.step-backward'),
        restart: document.querySelector('.restart'),
        progressBar: document.querySelector('.progress-bar'),
        progressIndicator: document.querySelector('.progress-bar-fill')
    };

    flowVisualizer.attachControls(controls);

    return flowVisualizer;
}

/**
 * Render network flow algorithm visualization
 */
export function renderNetworkFlowVisualization(data, speed = 5) {
    const visualizer = initNetworkFlowVisualization();
    if (!visualizer) return;

    // Initialize visualizer with data
    visualizer.initialize(data);

    // Start animation after a short delay
    setTimeout(() => {
        visualizer.startAnimation(speed);
    }, 100);
}

// Export the visualizers
export default {
    initNetworkFlowVisualization,
    renderNetworkFlowVisualization,
    NetworkFlowVisualizer
};