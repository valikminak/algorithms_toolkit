/**
 * A visualization system for string algorithms
 */
class StringVisualizer {
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
            text: {
                normal: '#333333',
                highlight: '#e04a6e',
                match: '#28a745',
                pattern: '#4a6ee0'
            },
            box: {
                outline: '#dddddd',
                highlight: '#ffeeee',
                match: '#eeffee',
                pattern: '#eeeeff'
            }
        };

        this.fontFamily = 'monospace';
        this.fontSize = 18;
        this.cellSize = 30;
        this.patternOffsetY = 120;
        this.textOffsetY = 50;

        // Bind methods
        this.render = this.render.bind(this);
        this.animate = this.animate.bind(this);
        this.handleResize = this.handleResize.bind(this);

        // Set up event listeners
        window.addEventListener('resize', this.handleResize);
    }

    /**
     * Initialize the visualizer with string algorithm data
     * @param {Object} data - The visualization data
     */
    initialize(data) {
        this.reset();

        this.algorithm = data.algorithm || '';
        this.text = data.text || '';
        this.pattern = data.pattern || '';
        this.frames = data.steps || [];
        this.lps = data.lps || [];
        this.matches = data.matches || [];

        this.currentFrame = 0;

        this.adjustCanvasSize();
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
     * Adjust canvas size based on content
     */
    adjustCanvasSize() {
        const containerWidth = this.canvas.parentElement.clientWidth;
        const textLength = Math.max(this.text.length, 30);

        this.cellSize = Math.min(30, Math.floor((containerWidth - 40) / textLength));
        this.fontSize = Math.max(12, Math.floor(this.cellSize * 0.7));

        const canvasWidth = Math.max(containerWidth, this.cellSize * textLength + 40);
        const canvasHeight = 350; // Fixed height

        this.canvas.width = canvasWidth;
        this.canvas.height = canvasHeight;
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
        if (this.algorithm === 'kmp') {
            this.renderKMPFrame(frame);
        } else if (this.algorithm === 'rabin_karp') {
            this.renderRabinKarpFrame(frame);
        } else if (this.algorithm === 'suffix_tree') {
            this.renderSuffixTreeFrame(frame);
        } else {
            this.renderGenericStringFrame(frame);
        }

        // Draw frame info
        this.drawFrameInfo(frame.info || '', this.currentFrame + 1, this.frames.length);
    }

    /**
     * Render a KMP algorithm frame
     */
    renderKMPFrame(frame) {
        // Display text with text_pos and pattern_pos highlighted
        this.drawText(this.text, 20, this.textOffsetY);

        // Draw pattern at current position
        if (frame.text_pos !== undefined && frame.pattern_pos !== undefined) {
            const patternX = 20 + (frame.text_pos - frame.pattern_pos) * this.cellSize;
            this.drawPattern(this.pattern, patternX, this.patternOffsetY, frame.pattern_pos);
        }

        // Draw LPS array if available
        if (this.lps && this.lps.length > 0) {
            this.drawLPSArray(this.lps, 20, 200);
        }

        // Draw matches
        if (frame.matches_so_far && frame.matches_so_far.length > 0) {
            this.drawMatches(frame.matches_so_far);
        }
    }

    /**
     * Render a Rabin-Karp algorithm frame
     */
    renderRabinKarpFrame(frame) {
        // Display text
        this.drawText(this.text, 20, this.textOffsetY);

        // Draw pattern at current position
        if (frame.position !== undefined) {
            const patternX = 20 + frame.position * this.cellSize;
            this.drawPattern(this.pattern, patternX, this.patternOffsetY);

            // Highlight text window
            this.highlightTextWindow(frame.position, frame.position + this.pattern.length - 1);
        }

        // Display hash values
        if (frame.pattern_hash !== undefined && frame.text_hash !== undefined) {
            const hashMatch = frame.hash_match ? '✓' : '✗';
            const stringMatch = frame.string_match ? '✓' : '✗';

            this.ctx.font = `${this.fontSize}px ${this.fontFamily}`;
            this.ctx.fillStyle = this.colors.text.normal;
            this.ctx.textAlign = 'left';

            const hashText = `Hash values: Pattern = ${frame.pattern_hash}, Window = ${frame.text_hash} (${hashMatch})`;
            const matchText = `String comparison: ${stringMatch}`;

            this.ctx.fillText(hashText, 20, 180);
            this.ctx.fillText(matchText, 20, 210);
        }

        // Draw matches
        if (frame.matches_so_far && frame.matches_so_far.length > 0) {
            this.drawMatches(frame.matches_so_far);
        }
    }

    /**
     * Render a suffix tree algorithm frame
     */
    renderSuffixTreeFrame(frame) {
        // For suffix tree, we'll show the current suffix and operations
        this.ctx.font = `${this.fontSize}px ${this.fontFamily}`;
        this.ctx.fillStyle = this.colors.text.normal;
        this.ctx.textAlign = 'left';

        // Draw text
        this.drawText(this.text, 20, this.textOffsetY);

        // Draw current suffix
        if (frame.suffix) {
            this.ctx.fillText(`Suffix: ${frame.suffix}`, 20, 120);

            // Highlight the suffix position in the text
            if (frame.position !== undefined) {
                this.highlightTextWindow(frame.position, this.text.length - 1);
            }
        }

        // Draw operations
        if (frame.operations && frame.operations.length > 0) {
            let y = 150;
            frame.operations.forEach(op => {
                let opText = '';
                switch (op.type) {
                    case 'traverse':
                        opText = `Traversing edge "${op.edge}"`;
                        break;
                    case 'split':
                        opText = `Splitting edge "${op.edge}" after ${op.match_length} characters`;
                        break;
                    case 'new_leaf':
                        opText = `Creating new leaf with edge "${op.edge}"`;
                        break;
                }

                this.ctx.fillText(opText, 20, y);
                y += 25;
            });
        }
    }

    /**
     * Render a generic string algorithm frame
     */
    renderGenericStringFrame(frame) {
        // Display text
        this.drawText(this.text, 20, this.textOffsetY);

        // Draw pattern if available
        if (this.pattern) {
            this.drawPattern(this.pattern, 20, this.patternOffsetY);
        }

        // Handle any special highlighting from the frame
        if (frame.highlights) {
            frame.highlights.forEach(highlight => {
                this.highlightTextWindow(highlight.start, highlight.end, highlight.color);
            });
        }
    }

    /**
     * Draw the text string with optional highlighting
     */
    drawText(text, x, y, highlightIndex = -1) {
        this.ctx.font = `${this.fontSize}px ${this.fontFamily}`;

        // Draw boxes first
        for (let i = 0; i < text.length; i++) {
            const cellX = x + i * this.cellSize;
            this.ctx.fillStyle = (i === highlightIndex) ?
                this.colors.box.highlight : this.colors.box.outline;
            this.ctx.fillRect(cellX, y - this.fontSize, this.cellSize, this.fontSize * 1.5);
            this.ctx.strokeStyle = '#888';
            this.ctx.strokeRect(cellX, y - this.fontSize, this.cellSize, this.fontSize * 1.5);
        }

        // Draw text on top of boxes
        for (let i = 0; i < text.length; i++) {
            const cellX = x + i * this.cellSize;
            this.ctx.fillStyle = (i === highlightIndex) ?
                this.colors.text.highlight : this.colors.text.normal;
            this.ctx.textAlign = 'center';
            this.ctx.fillText(text[i], cellX + this.cellSize / 2, y);
        }

        // Draw indices below the text
        this.ctx.fillStyle = '#888';
        this.ctx.font = `${Math.floor(this.fontSize * 0.7)}px ${this.fontFamily}`;
        for (let i = 0; i < text.length; i++) {
            const cellX = x + i * this.cellSize;
            this.ctx.fillText(i.toString(), cellX + this.cellSize / 2, y + 20);
        }
    }

    /**
     * Draw the pattern string with optional highlighting at a specific position
     */
    drawPattern(pattern, x, y, highlightIndex = -1) {
        this.ctx.font = `${this.fontSize}px ${this.fontFamily}`;

        // Draw boxes first
        for (let i = 0; i < pattern.length; i++) {
            const cellX = x + i * this.cellSize;
            this.ctx.fillStyle = (i === highlightIndex) ?
                this.colors.box.highlight : this.colors.box.pattern;
            this.ctx.fillRect(cellX, y - this.fontSize, this.cellSize, this.fontSize * 1.5);
            this.ctx.strokeStyle = '#888';
            this.ctx.strokeRect(cellX, y - this.fontSize, this.cellSize, this.fontSize * 1.5);
        }

        // Draw text on top of boxes
        for (let i = 0; i < pattern.length; i++) {
            const cellX = x + i * this.cellSize;
            this.ctx.fillStyle = (i === highlightIndex) ?
                this.colors.text.highlight : this.colors.text.pattern;
            this.ctx.textAlign = 'center';
            this.ctx.fillText(pattern[i], cellX + this.cellSize / 2, y);
        }
    }

    /**
     * Highlight a window of text
     */
    highlightTextWindow(start, end, color = this.colors.box.highlight) {
        const x = 20 + start * this.cellSize;
        const width = (end - start + 1) * this.cellSize;

        // Draw semi-transparent highlight
        this.ctx.fillStyle = color;
        this.ctx.globalAlpha = 0.3;
        this.ctx.fillRect(x, this.textOffsetY - this.fontSize, width, this.fontSize * 1.5);
        this.ctx.globalAlpha = 1.0;

        // Draw border
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(x, this.textOffsetY - this.fontSize, width, this.fontSize * 1.5);
    }

    /**
     * Draw the LPS (Longest Prefix which is also Suffix) array for KMP
     */
    drawLPSArray(lps, x, y) {
        this.ctx.font = `${this.fontSize}px ${this.fontFamily}`;
        this.ctx.fillStyle = this.colors.text.normal;
        this.ctx.textAlign = 'left';
        this.ctx.fillText("LPS Array:", x, y - 30);

        // Draw pattern character boxes
        for (let i = 0; i < lps.length; i++) {
            const cellX = x + i * this.cellSize;

            // Draw pattern character
            this.ctx.fillStyle = this.colors.box.pattern;
            this.ctx.fillRect(cellX, y - this.fontSize, this.cellSize, this.fontSize * 1.5);
            this.ctx.strokeStyle = '#888';
            this.ctx.strokeRect(cellX, y - this.fontSize, this.cellSize, this.fontSize * 1.5);

            this.ctx.fillStyle = this.colors.text.pattern;
            this.ctx.textAlign = 'center';
            this.ctx.fillText(this.pattern[i], cellX + this.cellSize / 2, y);
        }

        // Draw LPS values
        for (let i = 0; i < lps.length; i++) {
            const cellX = x + i * this.cellSize;

            // Draw LPS value
            this.ctx.fillStyle = '#f8f8f8';
            this.ctx.fillRect(cellX, y + 10, this.cellSize, this.fontSize * 1.5);
            this.ctx.strokeStyle = '#888';
            this.ctx.strokeRect(cellX, y + 10, this.cellSize, this.fontSize * 1.5);

            this.ctx.fillStyle = this.colors.text.normal;
            this.ctx.textAlign = 'center';
            this.ctx.fillText(lps[i].toString(), cellX + this.cellSize / 2, y + 28);
        }

        // Draw indices
        this.ctx.fillStyle = '#888';
        this.ctx.font = `${Math.floor(this.fontSize * 0.7)}px ${this.fontFamily}`;
        for (let i = 0; i < lps.length; i++) {
            const cellX = x + i * this.cellSize;
            this.ctx.fillText(i.toString(), cellX + this.cellSize / 2, y - 25);
        }
    }

    /**
     * Draw matches found so far
     */
    drawMatches(matches) {
        if (!matches.length) return;

        this.ctx.font = `${this.fontSize}px ${this.fontFamily}`;
        this.ctx.fillStyle = this.colors.text.match;
        this.ctx.textAlign = 'left';

        // Draw matches heading
        this.ctx.fillText("Matches found:", 20, 250);

        // Draw each match position
        const matchesText = matches.map(pos => pos.toString()).join(', ');
        this.ctx.fillText(matchesText, 20, 280);

        // Highlight matches in the text
        matches.forEach(pos => {
            const x = 20 + pos * this.cellSize;
            const width = this.pattern.length * this.cellSize;

            // Draw semi-transparent highlight
            this.ctx.fillStyle = this.colors.box.match;
            this.ctx.globalAlpha = 0.3;
            this.ctx.fillRect(x, this.textOffsetY - this.fontSize, width, this.fontSize * 1.5);
            this.ctx.globalAlpha = 1.0;

            // Draw border
            this.ctx.strokeStyle = this.colors.text.match;
            this.ctx.strokeRect(x, this.textOffsetY - this.fontSize, width, this.fontSize * 1.5);
        });
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
 * Register the string visualizer with the main visualization system
 */
export function initStringVisualization() {
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

    // Create string visualizer instance
    const stringVisualizer = new StringVisualizer('visualization-canvas');

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

    stringVisualizer.attachControls(controls);

    return stringVisualizer;
}

/**
 * Render string algorithm visualization
 */
export function renderStringVisualization(data, speed = 5) {
    const visualizer = initStringVisualization();
    if (!visualizer) return;

    // Extract visualization data based on algorithm type
    let visualizationData = {
        algorithm: data.algorithm,
        text: data.text || '',
        pattern: data.pattern || ''
    };

    if (data.algorithm === 'kmp') {
        visualizationData.steps = data.steps || [];
        visualizationData.lps = data.lps || [];
        visualizationData.matches = data.matches || [];
    } else if (data.algorithm === 'rabin_karp') {
        visualizationData.steps = data.steps || [];
        visualizationData.matches = data.matches || [];
    } else if (data.algorithm === 'suffix_tree') {
        visualizationData.steps = data.steps || [];
        visualizationData.suffixes = data.suffixes || [];
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
    initStringVisualization,
    renderStringVisualization,
    StringVisualizer
};