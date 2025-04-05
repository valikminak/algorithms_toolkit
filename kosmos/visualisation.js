/**
 * Base Visualizer class that provides common functionality for algorithm visualizations.
 * Algorithm-specific visualizers should extend this class.
 */
class VisualizerBase {
    /**
     * Create a base visualizer
     * @param {string} containerId - ID of the HTML container element
     * @param {Object} options - Visualization options
     */
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container element with ID '${containerId}' not found`);
        }

        this.options = {
            width: options.width || this.container.clientWidth || 800,
            height: options.height || 400,
            animationSpeed: options.animationSpeed || 5,
            fontSize: options.fontSize || 12,
            fontFamily: options.fontFamily || 'Arial, sans-serif',
            colors: options.colors || {
                background: '#f5f7fa',
                primary: '#4a6ee0',
                secondary: '#6e4ae0',
                highlight: '#e04a6e',
                text: '#333333',
                lightText: '#666666'
            },
            ...options
        };

        // Animation state
        this.frames = [];
        this.currentFrame = 0;
        this.isPlaying = false;
        this.animationTimer = null;

        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.options.width;
        this.canvas.height = this.options.height;
        this.canvas.className = 'visualization-canvas';
        this.ctx = this.canvas.getContext('2d');

        // Create info text element
        this.infoText = document.createElement('div');
        this.infoText.className = 'info-text';

        // Create control elements
        this.controls = this.createControls();

        // Add elements to container
        this.container.innerHTML = '';
        this.container.appendChild(this.canvas);
        this.container.appendChild(this.infoText);
        this.container.appendChild(this.controls);

        // Bind methods
        this.render = this.render.bind(this);
        this.play = this.play.bind(this);
        this.pause = this.pause.bind(this);
        this.next = this.next.bind(this);
        this.previous = this.previous.bind(this);
        this.goToFrame = this.goToFrame.bind(this);
        this.reset = this.reset.bind(this);

        // Handle window resize
        window.addEventListener('resize', () => {
            if (this.container.clientWidth) {
                this.canvas.width = this.container.clientWidth;
                this.render();
            }
        });
    }

    /**
     * Create animation controls
     * @returns {HTMLElement} The controls container
     */
    createControls() {
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'animation-controls';

        const controlsHTML = `
            <button class="restart" title="Restart">⏮️</button>
            <button class="step-backward" title="Previous Step">⏪</button>
            <button class="play-pause" title="Play/Pause">▶️</button>
            <button class="step-forward" title="Next Step">⏩</button>
            <div class="progress-bar">
                <div class="progress-bar-fill"></div>
            </div>
        `;

        controlsContainer.innerHTML = controlsHTML;

        // Add event listeners
        const playPauseBtn = controlsContainer.querySelector('.play-pause');
        const stepForwardBtn = controlsContainer.querySelector('.step-forward');
        const stepBackwardBtn = controlsContainer.querySelector('.step-backward');
        const restartBtn = controlsContainer.querySelector('.restart');
        const progressBar = controlsContainer.querySelector('.progress-bar');

        playPauseBtn.addEventListener('click', () => {
            if (this.isPlaying) {
                this.pause();
                playPauseBtn.textContent = '▶️';
            } else {
                this.play();
                playPauseBtn.textContent = '⏸️';
            }
        });

        stepForwardBtn.addEventListener('click', () => {
            this.pause();
            playPauseBtn.textContent = '▶️';
            this.next();
        });

        stepBackwardBtn.addEventListener('click', () => {
            this.pause();
            playPauseBtn.textContent = '▶️';
            this.previous();
        });

        restartBtn.addEventListener('click', () => {
            this.pause();
            playPauseBtn.textContent = '▶️';
            this.goToFrame(0);
        });

        progressBar.addEventListener('click', (e) => {
            const rect = progressBar.getBoundingClientRect();
            const clickPosition = e.clientX - rect.left;
            const percentage = clickPosition / rect.width;

            // Calculate frame index based on percentage
            const frameIndex = Math.floor(percentage * this.frames.length);
            this.goToFrame(frameIndex);

            this.pause();
            playPauseBtn.textContent = '▶️';
        });

        return controlsContainer;
    }

    /**
     * Set visualization data
     * @param {Array} frames - Array of visualization frames
     */
    setData(frames) {
        this.frames = frames || [];
        this.currentFrame = 0;

        // Update the progress bar
        this.updateProgressBar(0);

        // Render first frame
        if (this.frames.length > 0) {
            this.render();
        } else {
            this.showPlaceholder("No visualization data available");
        }
    }

    /**
     * Start playing the animation
     */
    play() {
        if (!this.frames.length) return;

        this.isPlaying = true;

        // Clear any existing timer
        if (this.animationTimer) {
            clearInterval(this.animationTimer);
        }

        // Calculate delay based on speed (1-10)
        const delay = 1000 / this.options.animationSpeed;

        // Start animation loop
        this.animationTimer = setInterval(() => {
            if (this.currentFrame >= this.frames.length - 1) {
                // Reached the end, stop the animation
                this.pause();
                const playPauseBtn = this.container.querySelector('.play-pause');
                if (playPauseBtn) {
                    playPauseBtn.textContent = '▶️';
                }
                return;
            }

            this.next();
        }, delay);
    }

    /**
     * Pause the animation
     */
    pause() {
        this.isPlaying = false;

        if (this.animationTimer) {
            clearInterval(this.animationTimer);
            this.animationTimer = null;
        }
    }

    /**
     * Go to the next frame
     */
    next() {
        if (this.currentFrame < this.frames.length - 1) {
            this.currentFrame++;
            this.render();
            this.updateProgressBar((this.currentFrame / (this.frames.length - 1)) * 100);
        }
    }

    /**
     * Go to the previous frame
     */
    previous() {
        if (this.currentFrame > 0) {
            this.currentFrame--;
            this.render();
            this.updateProgressBar((this.currentFrame / (this.frames.length - 1)) * 100);
        }
    }

    /**
     * Go to a specific frame
     * @param {number} frameIndex - Index of the frame to show
     */
    goToFrame(frameIndex) {
        if (frameIndex >= 0 && frameIndex < this.frames.length) {
            this.currentFrame = frameIndex;
            this.render();
            this.updateProgressBar((this.currentFrame / (this.frames.length - 1)) * 100);
        }
    }

    /**
     * Update the progress bar
     * @param {number} percentage - Percentage completion (0-100)
     */
    updateProgressBar(percentage) {
        const progressBar = this.container.querySelector('.progress-bar-fill');
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }
    }

    /**
     * Update the info text
     * @param {string} text - Text to display
     */
    updateInfoText(text) {
        if (this.infoText) {
            this.infoText.textContent = text || '';
        }
    }

    /**
     * Show a placeholder message when no visualization is available
     * @param {string} message - Message to display
     */
    showPlaceholder(message) {
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.fillStyle = this.options.colors.lightText;
            this.ctx.font = `16px ${this.options.fontFamily}`;
            this.ctx.textAlign = 'center';
            this.ctx.fillText(message, this.canvas.width / 2, this.canvas.height / 2);
        }

        this.updateInfoText('');
    }

    /**
     * Reset the visualization
     */
    reset() {
        this.pause();
        this.frames = [];
        this.currentFrame = 0;

        // Clear canvas
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }

        // Reset info text
        this.updateInfoText('');

        // Reset progress bar
        this.updateProgressBar(0);

        // Reset play/pause button
        const playPauseBtn = this.container.querySelector('.play-pause');
        if (playPauseBtn) {
            playPauseBtn.textContent = '▶️';
        }
    }

    /**
     * Render the current frame
     * This method should be overridden by subclasses
     */
    render() {
        if (!this.frames.length || this.currentFrame >= this.frames.length) {
            this.showPlaceholder("No data to visualize");
            return;
        }

        const frame = this.frames[this.currentFrame];

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw frame info
        if (frame.info) {
            this.updateInfoText(frame.info);
        } else {
            this.updateInfoText(`Frame ${this.currentFrame + 1} of ${this.frames.length}`);
        }

        // Subclasses should override this method to provide actual rendering
        this.renderPlaceholder();
    }

    /**
     * Render a placeholder when no specialized rendering is available
     */
    renderPlaceholder() {
        this.ctx.fillStyle = this.options.colors.background;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.fillStyle = this.options.colors.lightText;
        this.ctx.font = `16px ${this.options.fontFamily}`;
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            'Visualization not implemented for this algorithm type',
            this.canvas.width / 2,
            this.canvas.height / 2
        );
    }

    /**
     * Set the animation speed
     * @param {number} speed - Speed value (1-10)
     */
    setAnimationSpeed(speed) {
        this.options.animationSpeed = speed;

        // Restart animation if it's playing
        if (this.isPlaying) {
            this.pause();
            this.play();
        }
    }
}

// Export the class if in a module environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VisualizerBase;
} else if (typeof define === 'function' && define.amd) {
    define([], function () {
        return VisualizerBase;
    });
} else {
    window.VisualizerBase = VisualizerBase;
}