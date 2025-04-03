// graph-ui.js - UI components for graph algorithm visualization

/**
 * Graph UI component for creating and editing graph structures
 */
class GraphUI {
    constructor(containerEl) {
        this.container = containerEl;
        this.graph = {
            vertices: [],
            edges: [],
            directed: false,
            weighted: true
        };
        this.selectedVertex = null;
        this.selectedEdge = null;
        this.draggingVertex = null;
        this.startVertex = null;
        this.targetVertex = null;
        this.isCreatingEdge = false;

        // UI Elements
        this.canvas = null;
        this.ctx = null;
        this.controlsPanel = null;

        // Options
        this.nodeRadius = 20;
        this.edgeWidth = 2;
        this.colors = {
            background: '#f8f9fa',
            vertex: {
                default: '#4a6ee0',
                selected: '#e04a6e',
                start: '#9c27b0',
                target: '#ff5722',
                hover: '#6e4ae0'
            },
            edge: {
                default: '#666666',
                selected: '#e04a6e',
                hover: '#6e4ae0'
            },
            text: '#ffffff'
        };

        // Initialize the UI
        this.init();
    }

    /**
     * Initialize the graph UI
     */
    init() {
        // Create container structure
        this.container.innerHTML = `
            <div class="graph-editor">
                <div class="graph-canvas-container">
                    <canvas id="graph-canvas" width="800" height="400"></canvas>
                </div>
                <div class="graph-controls">
                    <div class="graph-actions">
                        <button id="add-vertex-btn" class="btn">Add Vertex</button>
                        <button id="delete-vertex-btn" class="btn">Delete Vertex</button>
                        <button id="add-edge-btn" class="btn">Add Edge</button>
                        <button id="delete-edge-btn" class="btn">Delete Edge</button>
                        <button id="clear-graph-btn" class="btn">Clear Graph</button>
                    </div>
                    <div class="graph-options">
                        <label>
                            <input type="checkbox" id="directed-graph-check"> Directed Graph
                        </label>
                        <label>
                            <input type="checkbox" id="weighted-graph-check" checked> Weighted Graph
                        </label>
                    </div>
                    <div class="algorithm-setup">
                        <div class="vertex-selector">
                            <label for="start-vertex-select">Start Vertex:</label>
                            <select id="start-vertex-select"></select>
                        </div>
                        <div class="vertex-selector">
                            <label for="target-vertex-select">Target Vertex:</label>
                            <select id="target-vertex-select"></select>
                        </div>
                    </div>
                    <div class="example-graphs">
                        <label for="example-graph-select">Load Example Graph:</label>
                        <select id="example-graph-select">
                            <option value="">Select an example</option>
                            <!-- Examples will be loaded here -->
                        </select>
                    </div>
                </div>
            </div>
        `;

        // Get references to UI elements
        this.canvas = document.getElementById('graph-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.controlsPanel = this.container.querySelector('.graph-controls');

        // Set up event listeners
        this.setupEventListeners();

        // Adjust canvas size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // Load example graphs
        this.loadExampleGraphs();

        // Initial render
        this.render();
    }

    /**
     * Set up all event listeners
     */
    setupEventListeners() {
        // Canvas event listeners for graph interaction
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('dblclick', this.handleDoubleClick.bind(this));

        // Button event listeners
        document.getElementById('add-vertex-btn').addEventListener('click', () => {
            this.isCreatingEdge = false;
            this.addVertex();
        });

        document.getElementById('delete-vertex-btn').addEventListener('click', () => {
            if (this.selectedVertex !== null) {
                this.removeVertex(this.selectedVertex);
            }
        });

        document.getElementById('add-edge-btn').addEventListener('click', () => {
            this.isCreatingEdge = true;
            this.selectedVertex = null;
            this.startVertex = null;
        });

        document.getElementById('delete-edge-btn').addEventListener('click', () => {
            if (this.selectedEdge !== null) {
                this.removeEdge(this.selectedEdge);
            }
        });

        document.getElementById('clear-graph-btn').addEventListener('click', () => {
            if (confirm('Are you sure you want to clear the graph?')) {
                this.clearGraph();
            }
        });

        // Option checkboxes
        document.getElementById('directed-graph-check').addEventListener('change', (e) => {
            this.graph.directed = e.target.checked;
            this.render();
        });

        document.getElementById('weighted-graph-check').addEventListener('change', (e) => {
            this.graph.weighted = e.target.checked;
            this.render();
        });

        // Vertex selectors for algorithms
        document.getElementById('start-vertex-select').addEventListener('change', (e) => {
            this.startVertex = e.target.value;
            this.render();
        });

        document.getElementById('target-vertex-select').addEventListener('change', (e) => {
            this.targetVertex = e.target.value;
            this.render();
        });

        // Example graph selector
        document.getElementById('example-graph-select').addEventListener('change', (e) => {
            if (e.target.value) {
                this.loadGraph(e.target.value);
            }
        });
    }

    /**
     * Handle mouse down event on canvas
     */
    handleMouseDown(e) {
        const pos = this.getCanvasPosition(e);

        // Check if clicking on a vertex
        const vertexIndex = this.findVertexAt(pos.x, pos.y);
        if (vertexIndex !== -1) {
            this.selectedVertex = vertexIndex;
            this.draggingVertex = vertexIndex;

            // If creating an edge, set as start or end vertex
            if (this.isCreatingEdge) {
                if (this.startVertex === null) {
                    this.startVertex = vertexIndex;
                } else {
                    // Create edge from startVertex to this vertex
                    this.addEdge(this.startVertex, vertexIndex);
                    this.isCreatingEdge = false;
                    this.startVertex = null;
                }
            }

            this.render();
            return;
        }

        // Check if clicking on an edge
        const edgeIndex = this.findEdgeAt(pos.x, pos.y);
        if (edgeIndex !== -1) {
            this.selectedVertex = null;
            this.selectedEdge = edgeIndex;
            this.render();
            return;
        }

        // Clicking on empty space
        this.selectedVertex = null;
        this.selectedEdge = null;
        this.render();
    }

    /**
     * Handle mouse move event on canvas
     */
    handleMouseMove(e) {
        const pos = this.getCanvasPosition(e);

        // If dragging a vertex, update its position
        if (this.draggingVertex !== null) {
            this.graph.vertices[this.draggingVertex].x = pos.x;
            this.graph.vertices[this.draggingVertex].y = pos.y;
            this.render();
        }
    }

    /**
     * Handle mouse up event on canvas
     */
    handleMouseUp() {
        this.draggingVertex = null;
    }

    /**
     * Handle double click event on canvas
     */
    handleDoubleClick(e) {
        const pos = this.getCanvasPosition(e);

        // Check if double-clicking on a vertex
        const vertexIndex = this.findVertexAt(pos.x, pos.y);
        if (vertexIndex !== -1) {
            // Prompt for new label
            const newLabel = prompt('Enter new vertex label:', this.graph.vertices[vertexIndex].label);
            if (newLabel !== null && newLabel.trim() !== '') {
                this.graph.vertices[vertexIndex].label = newLabel.trim();
                this.updateVertexSelectors();
                this.render();
            }
            return;
        }

        // Check if double-clicking on an edge
        const edgeIndex = this.findEdgeAt(pos.x, pos.y);
        if (edgeIndex !== -1 && this.graph.weighted) {
            // Prompt for new weight
            const edge = this.graph.edges[edgeIndex];
            const newWeight = prompt('Enter new edge weight:', edge.weight);
            if (newWeight !== null) {
                const weight = parseFloat(newWeight);
                if (!isNaN(weight)) {
                    edge.weight = weight;
                    this.render();
                }
            }
            return;
        }

        // Double-clicking on empty space adds a vertex
        this.addVertex(pos.x, pos.y);
    }

    /**
     * Get mouse position relative to canvas
     */
    getCanvasPosition(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    /**
     * Find the index of a vertex at the given position
     */
    findVertexAt(x, y) {
        for (let i = 0; i < this.graph.vertices.length; i++) {
            const vertex = this.graph.vertices[i];
            const dx = x - vertex.x;
            const dy = y - vertex.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance <= this.nodeRadius) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Find the index of an edge near the given position
     */
    findEdgeAt(x, y) {
        const threshold = 5; // Distance threshold for edge selection

        for (let i = 0; i < this.graph.edges.length; i++) {
            const edge = this.graph.edges[i];
            const sourceVertex = this.graph.vertices[edge.source];
            const targetVertex = this.graph.vertices[edge.target];

            // Calculate distance from point to line
            const distance = this.distanceToLine(
                x, y,
                sourceVertex.x, sourceVertex.y,
                targetVertex.x, targetVertex.y
            );

            if (distance <= threshold) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Calculate the distance from a point to a line segment
     */
    distanceToLine(x, y, x1, y1, x2, y2) {
        const A = x - x1;
        const B = y - y1;
        const C = x2 - x1;
        const D = y2 - y1;

        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;

        if (lenSq !== 0) {
            param = dot / lenSq;
        }

        let xx, yy;

        if (param < 0) {
            xx = x1;
            yy = y1;
        } else if (param > 1) {
            xx = x2;
            yy = y2;
        } else {
            xx = x1 + param * C;
            yy = y1 + param * D;
        }

        const dx = x - xx;
        const dy = y - yy;

        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Add a new vertex to the graph
     */
    addVertex(x, y) {
        // Generate a new label (A, B, C, ... Z, AA, AB, ...)
        const label = this.generateVertexLabel();

        // If x and y are not provided, place in the center of visible area
        if (x === undefined || y === undefined) {
            x = this.canvas.width / 2;
            y = this.canvas.height / 2;
        }

        this.graph.vertices.push({
            label: label,
            x: x,
            y: y
        });

        // Select the new vertex
        this.selectedVertex = this.graph.vertices.length - 1;
        this.selectedEdge = null;

        // Update vertex selectors
        this.updateVertexSelectors();

        this.render();
    }

    /**
     * Remove a vertex from the graph
     */
    removeVertex(index) {
        // Remove all edges connected to this vertex
        this.graph.edges = this.graph.edges.filter(edge =>
            edge.source !== index && edge.target !== index
        );

        // Remove the vertex
        this.graph.vertices.splice(index, 1);

        // Update indices in edges
        this.graph.edges.forEach(edge => {
            if (edge.source > index) edge.source--;
            if (edge.target > index) edge.target--;
        });

        // Update selection
        this.selectedVertex = null;
        this.selectedEdge = null;

        // Update vertex selectors
        this.updateVertexSelectors();

        this.render();
    }

    /**
     * Add a new edge to the graph
     */
    addEdge(sourceIndex, targetIndex) {
        // Don't add edge if source and target are the same
        if (sourceIndex === targetIndex) return;

        // Check if edge already exists
        const edgeExists = this.graph.edges.some(edge =>
            (edge.source === sourceIndex && edge.target === targetIndex) ||
            (!this.graph.directed && edge.source === targetIndex && edge.target === sourceIndex)
        );

        if (!edgeExists) {
            const weight = this.graph.weighted ? 1 : null;

            this.graph.edges.push({
                source: sourceIndex,
                target: targetIndex,
                weight: weight
            });

            this.selectedEdge = this.graph.edges.length - 1;
            this.selectedVertex = null;

            this.render();
        }
    }

    /**
     * Remove an edge from the graph
     */
    removeEdge(index) {
        this.graph.edges.splice(index, 1);
        this.selectedEdge = null;
        this.render();
    }

    /**
     * Clear the entire graph
     */
    clearGraph() {
        this.graph.vertices = [];
        this.graph.edges = [];
        this.selectedVertex = null;
        this.selectedEdge = null;
        this.startVertex = null;
        this.targetVertex = null;

        // Update vertex selectors
        this.updateVertexSelectors();

        this.render();
    }

    /**
     * Generate a new vertex label
     */
    generateVertexLabel() {
        const labels = this.graph.vertices.map(v => v.label);
        let label = 'A';

        while (labels.includes(label)) {
            // Increment label (A -> B -> ... -> Z -> AA -> AB -> ...)
            let lastChar = label.charAt(label.length - 1);

            if (lastChar === 'Z') {
                label = label.substring(0, label.length - 1) + 'AA';
            } else {
                label = label.substring(0, label.length - 1) + String.fromCharCode(lastChar.charCodeAt(0) + 1);
            }
        }

        return label;
    }

    /**
     * Update the vertex selector dropdowns
     */
    updateVertexSelectors() {
        const startSelect = document.getElementById('start-vertex-select');
        const targetSelect = document.getElementById('target-vertex-select');

        // Save current selections
        const currentStart = startSelect.value;
        const currentTarget = targetSelect.value;

        // Clear options
        startSelect.innerHTML = '';
        targetSelect.innerHTML = '';

        // Add empty option
        startSelect.innerHTML = '<option value="">Select Start</option>';
        targetSelect.innerHTML = '<option value="">Select Target</option>';

        // Add options for each vertex
        this.graph.vertices.forEach((vertex, index) => {
            const startOption = document.createElement('option');
            startOption.value = index;
            startOption.textContent = vertex.label;
            startSelect.appendChild(startOption);

            const targetOption = document.createElement('option');
            targetOption.value = index;
            targetOption.textContent = vertex.label;
            targetSelect.appendChild(targetOption);
        });

        // Restore selections if vertices still exist
        if (currentStart && this.graph.vertices[currentStart]) {
            startSelect.value = currentStart;
            this.startVertex = parseInt(currentStart);
        } else {
            this.startVertex = null;
        }

        if (currentTarget && this.graph.vertices[currentTarget]) {
            targetSelect.value = currentTarget;
            this.targetVertex = parseInt(currentTarget);
        } else {
            this.targetVertex = null;
        }
    }

    /**
     * Load example graphs from the API
     */
    loadExampleGraphs() {
        fetch('/api/graph/example-graphs')
            .then(response => response.json())
            .then(examples => {
                const select = document.getElementById('example-graph-select');

                examples.forEach(example => {
                    const option = document.createElement('option');
                    option.value = example.id;
                    option.textContent = example.name;
                    select.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error loading example graphs:', error);
            });
    }

    /**
     * Load a specific example graph
     */
    loadGraph(graphId) {
        fetch(`/api/graph/example-graphs`)
            .then(response => response.json())
            .then(examples => {
                const example = examples.find(ex => ex.id === graphId);
                if (example) {
                    this.importGraph(example);
                }
            })
            .catch(error => {
                console.error('Error loading graph:', error);
            });
    }

    /**
     * Import a graph from JSON data
     */
    importGraph(graphData) {
        // Clear current graph
        this.clearGraph();

        // Set graph properties
        this.graph.directed = graphData.directed || false;
        this.graph.weighted = graphData.weighted || false;

        // Update checkboxes
        document.getElementById('directed-graph-check').checked = this.graph.directed;
        document.getElementById('weighted-graph-check').checked = this.graph.weighted;

        // Create vertices
        const vertexMap = new Map(); // Map vertex IDs to indices

        graphData.vertices.forEach((vertexId, index) => {
            // Create vertex with random position or layout if provided
            const x = Math.random() * (this.canvas.width - 100) + 50;
            const y = Math.random() * (this.canvas.height - 100) + 50;

            this.graph.vertices.push({
                label: vertexId,
                x: x,
                y: y
            });

            vertexMap.set(vertexId, index);
        });

        // Create edges
        graphData.edges.forEach(edge => {
            const sourceIndex = vertexMap.get(edge.source);
            const targetIndex = vertexMap.get(edge.target);

            if (sourceIndex !== undefined && targetIndex !== undefined) {
                this.graph.edges.push({
                    source: sourceIndex,
                    target: targetIndex,
                    weight: edge.weight || 1
                });
            }
        });

        // Apply force-directed layout
        this.applyForceLayout();

        // Update vertex selectors
        this.updateVertexSelectors();

        // Render the graph
        this.render();
    }

    /**
     * Export the current graph to JSON
     */
    exportGraph() {
        const result = {
            vertices: this.graph.vertices.map(v => v.label),
            edges: this.graph.edges.map(e => ({
                source: this.graph.vertices[e.source].label,
                target: this.graph.vertices[e.target].label,
                weight: e.weight
            })),
            directed: this.graph.directed,
            weighted: this.graph.weighted
        };

        return result;
    }

    /**
     * Apply a force-directed layout to the graph
     */
    applyForceLayout() {
        // Simple force-directed layout
        const iterations = 100;
        const repulsionForce = 10000;
        const attractionForce = 0.05;

        for (let i = 0; i < iterations; i++) {
            // Apply repulsive forces between all vertices
            for (let j = 0; j < this.graph.vertices.length; j++) {
                for (let k = j + 1; k < this.graph.vertices.length; k++) {
                    const v1 = this.graph.vertices[j];
                    const v2 = this.graph.vertices[k];

                    const dx = v2.x - v1.x;
                    const dy = v2.y - v1.y;
                    const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                    // Repulsive force (inverse square law)
                    const force = repulsionForce / (distance * distance);
                    const fx = dx / distance * force;
                    const fy = dy / distance * force;

                    v1.x -= fx;
                    v1.y -= fy;
                    v2.x += fx;
                    v2.y += fy;
                }
            }

            // Apply attractive forces along edges
            this.graph.edges.forEach(edge => {
                const v1 = this.graph.vertices[edge.source];
                const v2 = this.graph.vertices[edge.target];

                const dx = v2.x - v1.x;
                const dy = v2.y - v1.y;
                const distance = Math.sqrt(dx * dx + dy * dy) || 1;

                // Attractive force (proportional to distance)
                const force = distance * attractionForce;
                const fx = dx / distance * force;
                const fy = dy / distance * force;

                v1.x += fx;
                v1.y += fy;
                v2.x -= fx;
                v2.y -= fy;
            });

            // Keep vertices within canvas bounds
            this.graph.vertices.forEach(vertex => {
                vertex.x = Math.max(this.nodeRadius, Math.min(this.canvas.width - this.nodeRadius, vertex.x));
                vertex.y = Math.max(this.nodeRadius, Math.min(this.canvas.height - this.nodeRadius, vertex.y));
            });
        }
    }

    /**
     * Resize the canvas to fit its container
     */
    resizeCanvas() {
        const container = this.canvas.parentElement;
        if (container) {
            const width = container.clientWidth;
            const height = 400; // Fixed height or you can make it dynamic

            this.canvas.width = width;
            this.canvas.height = height;

            this.render();
        }
    }

    /**
     * Render the graph
     */
    render() {
        if (!this.ctx) return;

        // Clear the canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw background
        this.ctx.fillStyle = this.colors.background;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw edges
        this.graph.edges.forEach((edge, index) => {
            const sourceVertex = this.graph.vertices[edge.source];
            const targetVertex = this.graph.vertices[edge.target];

            // Determine edge color
            let color = this.colors.edge.default;
            if (index === this.selectedEdge) {
                color = this.colors.edge.selected;
            }

            this.drawEdge(
                sourceVertex.x, sourceVertex.y,
                targetVertex.x, targetVertex.y,
                color,
                this.graph.directed,
                edge.weight
            );
        });

        // Draw vertices
        this.graph.vertices.forEach((vertex, index) => {
            // Determine vertex color
            let color = this.colors.vertex.default;
            if (index === this.selectedVertex) {
                color = this.colors.vertex.selected;
            } else if (index === this.startVertex) {
                color = this.colors.vertex.start;
            } else if (index === this.targetVertex) {
                color = this.colors.vertex.target;
            }

            this.drawVertex(vertex.x, vertex.y, vertex.label, color);
        });

        // Draw edge being created if applicable
        if (this.isCreatingEdge && this.startVertex !== null) {
            const sourceVertex = this.graph.vertices[this.startVertex];
            const mousePos = this.lastMousePos || { x: sourceVertex.x + 100, y: sourceVertex.y };

            this.ctx.beginPath();
            this.ctx.moveTo(sourceVertex.x, sourceVertex.y);
            this.ctx.lineTo(mousePos.x, mousePos.y);
            this.ctx.strokeStyle = this.colors.edge.hover;
            this.ctx.lineWidth = this.edgeWidth;
            this.ctx.stroke();
        }
    }

    /**
     * Draw a vertex (node) on the canvas
     */
    drawVertex(x, y, label, color) {
        // Draw circle
        this.ctx.beginPath();
        this.ctx.arc(x, y, this.nodeRadius, 0, 2 * Math.PI);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Draw label
        this.ctx.fillStyle = this.colors.text;
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(label, x, y);
    }

    /**
     * Draw an edge on the canvas
     */
    drawEdge(x1, y1, x2, y2, color, directed, weight) {
        // Calculate the start and end points (adjusted for node radius)
        const dx = x2 - x1;
        const dy = y2 - y1;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Normalize direction vector
        const nx = dx / distance;
        const ny = dy / distance;

        // Adjust start and end points to be on the node circles
        const startX = x1 + nx * this.nodeRadius;
        const startY = y1 + ny * this.nodeRadius;
        const endX = x2 - nx * this.nodeRadius;
        const endY = y2 - ny * this.nodeRadius;

        // Draw the edge
        this.ctx.beginPath();
        this.ctx.moveTo(startX, startY);
        this.ctx.lineTo(endX, endY);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = this.edgeWidth;
        this.ctx.stroke();

        // Draw arrow if directed
        if (directed) {
            const arrowSize = 10;
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
        }

        // Draw weight if weighted
        if (weight !== null && weight !== undefined) {
            const midX = (startX + endX) / 2;
            const midY = (startY + endY) / 2;

            // Draw weight background
            this.ctx.fillStyle = 'white';
            this.ctx.beginPath();
            this.ctx.arc(midX, midY, 12, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 1;
            this.ctx.stroke();

            // Draw weight text
            this.ctx.fillStyle = '#333';
            this.ctx.font = '11px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(weight.toString(), midX, midY);
        }
    }

    /**
     * Get the current graph data
     */
    getGraphData() {
        return this.exportGraph();
    }

    /**
     * Get the current start and target vertices
     */
    getAlgorithmParams() {
        return {
            startVertex: this.startVertex !== null ? this.graph.vertices[this.startVertex].label : null,
            targetVertex: this.targetVertex !== null ? this.graph.vertices[this.targetVertex].label : null
        };
    }
}

// Export the GraphUI class
export default GraphUI;