// dp-visualization.js - Specialized visualization for dynamic programming algorithms

/**
 * A visualization system for dynamic programming algorithms
 */
class DPVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.frames = [];
        this.currentFrame = 0;
        this.animationTimer = null;
        this.paused = true;

        // Visualization properties
        this.colors = {
            background: '#f8f9fa',
            cell: {
                default: '#ffffff',
                active: '#e8f4ff',
                highlight: '#ffeeee',
                computed: '#e8ffe8',
                path: '#fff8e8'
            },
            text: {
                normal: '#333333',
                highlight: '#e04a6e',
                computed: '#28a745',
                header: '#4a6ee0'
            },
            border: {
                default: '#dddddd',
                highlight: '#e04a6e',
                computed: '#28a745'
            },
            gridLine: '#eeeeee'
        };

        this.cellSize = 40;
        this.padding = 20;
        this.fontFamily = 'Arial, sans-serif';
        this.fontSize = 14;

        // Bind methods
        this.render = this.render.bind(this);
        this.animate = this.animate.bind(this);
        this.handleResize = this.handleResize.bind(this);

        // Set up event listeners
        window.addEventListener('resize', this.handleResize);
    }

    /**
     * Initialize the visualizer with DP algorithm data
     * @param {Object} data - The visualization data
     */
    initialize(data) {
        this.reset();

        this.algorithm = data.algorithm || '';
        this.frames = data.steps || [];
        this.currentFrame = 0;

        // Extract DP-specific data
        this.dpTable = data.dp_table || [];
        this.parenthesisTable = data.parenthesis || [];
        this.problemData = this.extractProblemData(data);

        this.adjustCanvasSize();
        this.render();
    }

    /**
     * Extract relevant problem data from visualization data
     */
    extractProblemData(data) {
        const problemData = {};

        if (data.algorithm) {
            problemData.algorithm = data.algorithm;

            // Extract algorithm-specific data
            if (data.algorithm.includes('knapsack')) {
                problemData.weights = data.weights || [];
                problemData.values = data.values || [];
                problemData.capacity = data.capacity || 0;
                problemData.selected_items = data.selected_items || [];
            } else if (data.algorithm.includes('lcs')) {
                problemData.string1 = data.string1 || '';
                problemData.string2 = data.string2 || '';
                problemData.lcs = data.lcs || '';
            } else if (data.algorithm.includes('matrix_chain')) {
                problemData.dimensions = data.dimensions || [];
                problemData.parenthesization = data.parenthesization || '';
            }
        }

        return problemData;
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
     * Adjust canvas size based on DP table dimensions
     */
    adjustCanvasSize() {
        const containerWidth = this.canvas.parentElement.clientWidth;

        // Determine table dimensions
        let rows = 1, cols = 1;

        if (this.dpTable && this.dpTable.length > 0) {
            rows = this.dpTable.length;
            cols = Array.isArray(this.dpTable[0]) ? this.dpTable[0].length : 1;
        }

        // Calculate cell size based on available width
        this.cellSize = Math.max(30, Math.min(60, Math.floor((containerWidth - 100) / (cols + 1))));
        this.fontSize = Math.max(12, Math.floor(this.cellSize * 0.4));

        // Calculate canvas dimensions
        const width = Math.max(containerWidth, this.cellSize * (cols + 1) + this.padding * 2);
        const height = Math.max(400, this.cellSize * (rows + 2) + this.padding * 2 + 100);

        this.canvas.width = width;
        this.canvas.height = height;
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
        if (this.algorithm.includes('knapsack')) {
            this.renderKnapsackFrame(frame);
        } else if (this.algorithm.includes('lcs')) {
            this.renderLCSFrame(frame);
        } else if (this.algorithm.includes('matrix_chain')) {
            this.renderMatrixChainFrame(frame);
        } else {
            this.renderGenericDPFrame(frame);
        }

        // Draw frame info
        this.drawFrameInfo(frame.info || '', this.currentFrame + 1, this.frames.length);
    }

    /**
     * Render a knapsack algorithm frame
     */
    renderKnapsackFrame(frame) {
        const dpTable = frame.dp_table || this.dpTable;
        if (!dpTable || dpTable.length === 0) return;

        const rows = dpTable.length;
        const cols = dpTable[0].length;

        // Draw problem information
        this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
        this.ctx.fillStyle = this.colors.text.normal;
        this.ctx.textAlign = 'left';

        let infoY = this.padding;

        // Draw weights and values if available
        if (this.problemData.weights && this.problemData.values) {
            const weights = this.problemData.weights.join(', ');
            const values = this.problemData.values.join(', ');

            this.ctx.fillText(`Weights: [${weights}]`, this.padding, infoY);
            infoY += 25;

            this.ctx.fillText(`Values: [${values}]`, this.padding, infoY);
            infoY += 25;

            this.ctx.fillText(`Capacity: ${this.problemData.capacity}`, this.padding, infoY);
            infoY += 40;
        }

        // Draw the DP table
        const tableX = this.padding;
        const tableY = infoY;

        this.drawDPTable(dpTable, tableX, tableY, frame);

        // Draw selected items if available
        if (frame.selected_items && frame.selected_items.length > 0) {
            const selectedItemsY = tableY + (rows + 1) * this.cellSize + 30;

            this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
            this.ctx.fillStyle = this.colors.text.computed;
            this.ctx.textAlign = 'left';

            this.ctx.fillText('Selected items:', this.padding, selectedItemsY);
            this.ctx.fillText(frame.selected_items.join(', '), this.padding + 120, selectedItemsY);
        }
    }

    /**
     * Render an LCS (Longest Common Subsequence) algorithm frame
     */
    renderLCSFrame(frame) {
        const dpTable = frame.dp_table || this.dpTable;
        if (!dpTable || dpTable.length === 0) return;

        // Draw strings information
        this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
        this.ctx.fillStyle = this.colors.text.normal;
        this.ctx.textAlign = 'left';

        let infoY = this.padding;

        if (this.problemData.string1 && this.problemData.string2) {
            this.ctx.fillText(`String 1: "${this.problemData.string1}"`, this.padding, infoY);
            infoY += 25;

            this.ctx.fillText(`String 2: "${this.problemData.string2}"`, this.padding, infoY);
            infoY += 40;
        }

        // Draw the DP table with string characters as headers
        const tableX = this.padding;
        const tableY = infoY;

        this.drawLCSTable(dpTable, tableX, tableY, frame);

        // Draw LCS result if available
        if (frame.lcs_so_far) {
            const lcsY = tableY + (dpTable.length + 1) * this.cellSize + 30;

            this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
            this.ctx.fillStyle = this.colors.text.computed;
            this.ctx.textAlign = 'left';

            this.ctx.fillText('LCS so far:', this.padding, lcsY);
            this.ctx.fillText(`"${frame.lcs_so_far}"`, this.padding + 100, lcsY);
        }
    }

    /**
     * Render a Matrix Chain Multiplication algorithm frame
     */
    renderMatrixChainFrame(frame) {
        const dpTable = frame.dp_table || this.dpTable;
        if (!dpTable || dpTable.length === 0) return;

        // Draw problem information
        this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
        this.ctx.fillStyle = this.colors.text.normal;
        this.ctx.textAlign = 'left';

        let infoY = this.padding;

        if (this.problemData.dimensions) {
            this.ctx.fillText(`Matrix Dimensions: [${this.problemData.dimensions.join(', ')}]`, this.padding, infoY);
            infoY += 40;
        }

        // Draw the DP table
        const tableX = this.padding;
        const tableY = infoY;

        this.drawMatrixChainTable(dpTable, tableX, tableY, frame);

        // Draw parenthesization if available
        if (frame.parenthesis) {
            const parenthesisTable = frame.parenthesis;
            const parTableY = tableY + (dpTable.length + 1) * this.cellSize + 30;

            this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
            this.ctx.fillStyle = this.colors.text.header;
            this.ctx.textAlign = 'left';

            this.ctx.fillText('Parenthesization Table:', this.padding, parTableY - 10);

            // Draw a smaller parenthesization table
            this.drawParenthesisTable(parenthesisTable, tableX, parTableY, frame);
        }

        // Draw optimal parenthesization if available
        if (this.problemData.parenthesization) {
            const resultY = this.canvas.height - 60;

            this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
            this.ctx.fillStyle = this.colors.text.computed;
            this.ctx.textAlign = 'left';

            this.ctx.fillText('Optimal Parenthesization:', this.padding, resultY);
            this.ctx.fillText(this.problemData.parenthesization, this.padding + 200, resultY);
        }
    }

    /**
     * Render a generic DP algorithm frame
     */
    renderGenericDPFrame(frame) {
        const dpTable = frame.dp_table || this.dpTable;
        if (!dpTable || dpTable.length === 0) return;

        // Draw the DP table
        const tableX = this.padding;
        const tableY = this.padding + 40;

        this.drawDPTable(dpTable, tableX, tableY, frame);
    }

    /**
     * Draw a DP table with optional highlighting
     */
    drawDPTable(dpTable, x, y, frame) {
        const rows = dpTable.length;
        const cols = Array.isArray(dpTable[0]) ? dpTable[0].length : 1;

        // Draw table header
        this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
        this.ctx.fillStyle = this.colors.text.header;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        this.ctx.fillText('DP Table', x + (cols * this.cellSize) / 2 + this.cellSize / 2, y - 20);

        // Draw row headers (items)
        for (let i = 0; i < rows; i++) {
            // Determine cell color
            let cellColor = this.colors.cell.default;
            let textColor = this.colors.text.normal;
            let borderColor = this.colors.border.default;

            // Highlight if this is the current row in the frame
            if (frame && frame.i === i) {
                cellColor = this.colors.cell.highlight;
                borderColor = this.colors.border.highlight;
            }

            // Draw row header cell
            this.drawCell(x, y + (i + 1) * this.cellSize, this.cellSize, this.cellSize,
                         cellColor, borderColor, `Item ${i}`, textColor);
        }

        // Draw column headers (weights/capacities)
        for (let j = 0; j < cols; j++) {
            // Determine cell color
            let cellColor = this.colors.cell.default;
            let textColor = this.colors.text.normal;
            let borderColor = this.colors.border.default;

            // Highlight if this is the current column in the frame
            if (frame && frame.w === j) {
                cellColor = this.colors.cell.highlight;
                borderColor = this.colors.border.highlight;
            }

            // Draw column header cell
            this.drawCell(x + (j + 1) * this.cellSize, y, this.cellSize, this.cellSize,
                         cellColor, borderColor, `${j}`, textColor);
        }

        // Draw top-left cell
        this.drawCell(x, y, this.cellSize, this.cellSize,
                     this.colors.cell.default, this.colors.border.default, "i / w", this.colors.text.header);

        // Draw the DP table values
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                if (!Array.isArray(dpTable[i])) continue;

                // Determine cell color
                let cellColor = this.colors.cell.default;
                let textColor = this.colors.text.normal;
                let borderColor = this.colors.border.default;

                // Highlight active cell
                if (frame) {
                    // Current cell being computed
                    if (frame.i === i && frame.w === j) {
                        cellColor = this.colors.cell.active;
                        borderColor = this.colors.border.highlight;
                        textColor = this.colors.text.highlight;
                    }
                    // Item was included in the solution
                    else if (frame.included && frame.i === i && frame.w === j) {
                        cellColor = this.colors.cell.computed;
                        borderColor = this.colors.border.computed;
                        textColor = this.colors.text.computed;
                    }
                }

                // Draw cell
                const cellValue = dpTable[i][j];
                this.drawCell(x + (j + 1) * this.cellSize, y + (i + 1) * this.cellSize,
                             this.cellSize, this.cellSize, cellColor, borderColor,
                             cellValue !== undefined ? cellValue.toString() : '', textColor);
            }
        }
    }

    /**
     * Draw an LCS table with string characters as headers
     */
    drawLCSTable(dpTable, x, y, frame) {
        const rows = dpTable.length;
        const cols = dpTable[0].length;

        const string1 = this.problemData.string1 || '';
        const string2 = this.problemData.string2 || '';

        // Draw table header
        this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
        this.ctx.fillStyle = this.colors.text.header;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        this.ctx.fillText('LCS Table', x + (cols * this.cellSize) / 2, y - 20);

        // Draw row headers (string1 characters)
        this.drawCell(x, y, this.cellSize, this.cellSize,
                     this.colors.cell.default, this.colors.border.default, "", this.colors.text.header);

        for (let i = 0; i < rows - 1; i++) {
            const char = i < string1.length ? string1[i] : '';

            // Determine cell color
            let cellColor = this.colors.cell.default;
            let textColor = this.colors.text.normal;
            let borderColor = this.colors.border.default;

            // Highlight if this character is involved in a match
            if (frame && frame.i === i + 1) {
                cellColor = this.colors.cell.highlight;
                borderColor = this.colors.border.highlight;
            }

            // Draw row header cell
            this.drawCell(x, y + (i + 1) * this.cellSize, this.cellSize, this.cellSize,
                         cellColor, borderColor, char, textColor);
        }

        // Draw column headers (string2 characters)
        for (let j = 0; j < cols - 1; j++) {
            const char = j < string2.length ? string2[j] : '';

            // Determine cell color
            let cellColor = this.colors.cell.default;
            let textColor = this.colors.text.normal;
            let borderColor = this.colors.border.default;

            // Highlight if this character is involved in a match
            if (frame && frame.j === j + 1) {
                cellColor = this.colors.cell.highlight;
                borderColor = this.colors.border.highlight;
            }

            // Draw column header cell
            this.drawCell(x + (j + 1) * this.cellSize, y, this.cellSize, this.cellSize,
                         cellColor, borderColor, char, textColor);
        }

        // Draw the DP table values
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                // Determine cell color
                let cellColor = this.colors.cell.default;
                let textColor = this.colors.text.normal;
                let borderColor = this.colors.border.default;

                // Highlight active cell
                if (frame) {
                    // Current cell being computed
                    if (frame.i === i && frame.j === j) {
                        cellColor = this.colors.cell.active;
                        borderColor = this.colors.border.highlight;
                        textColor = this.colors.text.highlight;
                    }
                    // Character match
                    else if (frame.match && frame.i === i && frame.j === j) {
                        cellColor = this.colors.cell.computed;
                        borderColor = this.colors.border.computed;
                        textColor = this.colors.text.computed;
                    }
                }

                // Draw cell
                const cellValue = dpTable[i][j];
                this.drawCell(x + j * this.cellSize, y + i * this.cellSize,
                             this.cellSize, this.cellSize, cellColor, borderColor,
                             cellValue !== undefined ? cellValue.toString() : '', textColor);
            }
        }
    }

    /**
     * Draw a Matrix Chain table
     */
    drawMatrixChainTable(dpTable, x, y, frame) {
        const n = dpTable.length;

        // Draw table header
        this.ctx.font = `bold ${this.fontSize}px ${this.fontFamily}`;
        this.ctx.fillStyle = this.colors.text.header;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        this.ctx.fillText('Cost Table', x + (n * this.cellSize) / 2, y - 20);

        // Draw row and column headers (matrix indices)
        for (let i = 0; i < n; i++) {
            // Draw row header
            this.drawCell(x, y + (i + 1) * this.cellSize, this.cellSize, this.cellSize,
                         this.colors.cell.default, this.colors.border.default, `A${i}`, this.colors.text.header);

            // Draw column header
            this.drawCell(x + (i + 1) * this.cellSize, y, this.cellSize, this.cellSize,
                         this.colors.cell.default, this.colors.border.default, `A${i}`, this.colors.text.header);
        }

        // Draw top-left cell
        this.drawCell(x, y, this.cellSize, this.cellSize,
                     this.colors.cell.default, this.colors.border.default, "i / j", this.colors.text.header);

        // Draw the DP table values
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (j < i) {
                    // Draw empty cell for lower triangle (i > j)
                    this.drawCell(x + (j + 1) * this.cellSize, y + (i + 1) * this.cellSize,
                                 this.cellSize, this.cellSize, this.colors.cell.default,
                                 this.colors.border.default, "", this.colors.text.normal);
                    continue;
                }

                // Determine cell color
                let cellColor = this.colors.cell.default;
                let textColor = this.colors.text.normal;
                let borderColor = this.colors.border.default;

                // Highlight active cell
                if (frame) {
                    // Current cell being computed
                    if (frame.i === i && frame.j === j) {
                        cellColor = this.colors.cell.active;
                        borderColor = this.colors.border.highlight;
                        textColor = this.colors.text.highlight;
                    }
                    // Current chain length
                    else if (frame.l && j - i + 1 === frame.l) {
                        cellColor = this.colors.cell.highlight;
                    }
                }

                // Draw cell
                const cellValue = dpTable[i][j];
                this.drawCell(x + (j + 1) * this.cellSize, y + (i + 1) * this.cellSize,
                             this.cellSize, this.cellSize, cellColor, borderColor,
                             cellValue !== undefined && cellValue !== Infinity ? cellValue.toString() : '∞', textColor);
            }
        }
    }

    /**
     * Draw the parenthesization table for matrix chain multiplication
     */
    drawParenthesisTable(parenthesisTable, x, y, frame) {
        if (!parenthesisTable || parenthesisTable.length === 0) return;

        const n = parenthesisTable.length;
        const cellSize = this.cellSize * 0.8; // Smaller cells for parenthesis table

        // Draw row and column headers
        for (let i = 0; i < n; i++) {
            // Draw row header
            this.drawCell(x, y + (i + 1) * cellSize, cellSize, cellSize,
                         this.colors.cell.default, this.colors.border.default, `A${i}`, this.colors.text.header);

            // Draw column header
            this.drawCell(x + (i + 1) * cellSize, y, cellSize, cellSize,
                         this.colors.cell.default, this.colors.border.default, `A${i}`, this.colors.text.header);
        }

        // Draw top-left cell
        this.drawCell(x, y, cellSize, cellSize,
                     this.colors.cell.default, this.colors.border.default, "i / j", this.colors.text.header);

        // Draw the parenthesis table values
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (j < i) {
                    // Draw empty cell for lower triangle (i > j)
                    this.drawCell(x + (j + 1) * cellSize, y + (i + 1) * cellSize,
                                 cellSize, cellSize, this.colors.cell.default,
                                 this.colors.border.default, "", this.colors.text.normal);
                    continue;
                }

                // Determine cell color
                let cellColor = this.colors.cell.default;
                let textColor = this.colors.text.normal;
                let borderColor = this.colors.border.default;

                // Highlight active cell if it matches frame indices
                if (frame && frame.i === i && frame.j === j) {
                    cellColor = this.colors.cell.active;
                    borderColor = this.colors.border.highlight;
                }

                // Draw cell with split point (k value)
                const cellValue = parenthesisTable[i][j];
                this.drawCell(x + (j + 1) * cellSize, y + (i + 1) * cellSize,
                             cellSize, cellSize, cellColor, borderColor,
                             cellValue !== undefined ? `k=${cellValue}` : '', textColor);
            }
        }
    }

    /**
     * Draw a single cell in the table
     */
    drawCell(x, y, width, height, fillColor, strokeColor, text, textColor) {
        // Draw cell background
        this.ctx.fillStyle = fillColor;
        this.ctx.fillRect(x, y, width, height);

        // Draw cell border
        this.ctx.strokeStyle = strokeColor;
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(x, y, width, height);

        // Draw cell text
        this.ctx.fillStyle = textColor;
        this.ctx.font = `${this.fontSize}px ${this.fontFamily}`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(text, x + width / 2, y + height / 2);
    }

    /**
     * Draw frame information text
     */
    drawFrameInfo(info, currentFrame, totalFrames) {
        // Draw frame progress
        this.ctx.fillStyle = '#333';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'top';
        this.ctx.fillText(`Step ${currentFrame} of ${totalFrames}`, 10, 10);

        // Draw info text
        this.ctx.fillStyle = '#333';
        this.ctx.font = '14px Arial';
        this.ctx.textAlign = 'center';

        // Handle multi-line text if needed
        if (info.length > 80) {
            const words = info.split(' ');
            let line = '';
            let y = this.canvas.height - 60;
            const maxWidth = this.canvas.width - 40;
            const lineHeight = 20;

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
            this.ctx.fillText(info, this.canvas.width / 2, this.canvas.height - 40);
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
        this.adjustCanvasSize();
        this.render();
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
 * Register the DP visualizer with the main visualization system
 */
export function initDPVisualization() {
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

    // Create DP visualizer instance
    const dpVisualizer = new DPVisualizer('visualization-canvas');

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

    dpVisualizer.attachControls(controls);

    return dpVisualizer;
}

/**
 * Render DP algorithm visualization
 */
export function renderDPVisualization(data, speed = 5) {
    const visualizer = initDPVisualization();
    if (!visualizer) return;

    // Extract visualization data based on algorithm type
    let visualizationData = {
        algorithm: data.algorithm,
        steps: data.steps || [],
        dp_table: data.dp_table || []
    };

    // Add algorithm-specific data
    if (data.algorithm.includes('knapsack')) {
        visualizationData.weights = data.weights || [];
        visualizationData.values = data.values || [];
        visualizationData.capacity = data.capacity || 0;
        visualizationData.selected_items = data.selected_items || [];
    } else if (data.algorithm.includes('lcs')) {
        visualizationData.string1 = data.string1 || '';
        visualizationData.string2 = data.string2 || '';
        visualizationData.lcs = data.lcs || '';
    } else if (data.algorithm.includes('matrix_chain')) {
        visualizationData.dimensions = data.dimensions || [];
        visualizationData.parenthesization = data.parenthesization || '';
        visualizationData.parenthesis = data.parenthesis || [];
    }

    // Initialize visualizer with data
    visualizer.initialize(visualizationData);

    // Start animation after a short delay
    setTimeout(() => {
        visualizer.startAnimation(speed);
    }, 100);
}

// Export the visualizers
export default {
    initDPVisualization,
    renderDPVisualization,
    DPVisualizer
};