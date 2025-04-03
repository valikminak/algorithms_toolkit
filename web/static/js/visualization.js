// visualization.js - Handles all visualization rendering using Canvas

// Import graph visualizer
import GraphVisualizer from './graph-visualization.js';

let visualizationContainer;
let canvas;
let ctx;
let animationTimer;
let currentFrame = 0;
let frames = [];
let animationPaused = false;
let graphVisualizer = null;

// Initialize the visualization container and canvas
export function initVisualization() {
    visualizationContainer = document.getElementById('visualization-container');

    // Create canvas element if it doesn't exist
    canvas = document.createElement('canvas');
    canvas.id = 'visualization-canvas';
    canvas.width = visualizationContainer.clientWidth || 800;
    canvas.height = 400;
    canvas.className = 'visualization-canvas';

    // Clear container and add canvas
    visualizationContainer.innerHTML = '';
    visualizationContainer.appendChild(canvas);

    // Get 2D context
    ctx = canvas.getContext('2d');

    // Create info text element
    const infoText = document.createElement('div');
    infoText.className = 'info-text';
    infoText.id = 'info-text';
    visualizationContainer.appendChild(infoText);

    // Add animation controls
    const controls = createAnimationControls();
    visualizationContainer.appendChild(controls);

    // Initialize graph visualizer
    graphVisualizer = new GraphVisualizer('visualization-canvas');

    // Add window resize listener to make canvas responsive
    window.addEventListener('resize', () => {
        if (visualizationContainer.clientWidth) {
            canvas.width = visualizationContainer.clientWidth;
            if (frames.length > 0) {
                renderCurrentFrame();
            }
        }
    });
}

// Render visualization data
export function renderVisualization(data, speed) {
    clearVisualization();

    if (!data || !data.visualization || !Array.isArray(data.visualization) || data.visualization.length === 0) {
        showPlaceholder('No visualization data available');
        return;
    }

    // Store visualization frames
    frames = data.visualization;

    // Determine visualization type
    const category = data.category || determineCategory(data);

    // For graph algorithms, use the specialized graph visualizer
    if (category === 'graph') {
        renderGraphVisualization(data, speed);
        return;
    }

    // Create render function based on category
    let renderFunction;

    switch (category) {
        case 'sorting':
            renderFunction = renderSortingFrame;
            break;
        case 'searching':
            renderFunction = renderSearchingFrame;
            break;
        default:
            renderFunction = renderGenericFrame;
    }

    // Reset animation state
    currentFrame = 0;
    animationPaused = false;

    // Start animation
    animateFrames(renderFunction, speed);

    // Update controls event listeners
    setupControlListeners(renderFunction, speed);
}

// Render graph visualization using the graph visualizer
function renderGraphVisualization(data, speed) {
    if (!graphVisualizer) {
        console.error('Graph visualizer not initialized');
        return;
    }

    // Initialize graph visualizer with data
    graphVisualizer.initialize(data);

    // Set up controls for graph visualizer
    const controls = {
        playPause: document.querySelector('.play-pause'),
        next: document.querySelector('.step-forward'),
        prev: document.querySelector('.step-backward'),
        restart: document.querySelector('.restart'),
        progressBar: document.querySelector('.progress-bar'),
        progressIndicator: document.querySelector('.progress-bar-fill')
    };

    // Attach controls
    graphVisualizer.attachControls(controls);

    // Set animation speed
    graphVisualizer.setSpeed(speed);

    // Start animation
    setTimeout(() => {
        graphVisualizer.startAnimation(speed);
    }, 100);
}

// Clear the visualization and stop any animations
export function clearVisualization() {
    if (animationTimer) {
        clearInterval(animationTimer);
        animationTimer = null;
    }

    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Reset graph visualizer if it exists
    if (graphVisualizer) {
        graphVisualizer.reset();
    }

    const infoText = document.getElementById('info-text');
    if (infoText) {
        infoText.textContent = '';
    }

    currentFrame = 0;
    frames = [];
    animationPaused = false;

    // Reset progress bar
    const progressBar = document.querySelector('.progress-bar-fill');
    if (progressBar) {
        progressBar.style.width = '0%';
    }

    // Reset play/pause button
    const playPauseBtn = document.querySelector('.play-pause');
    if (playPauseBtn) {
        playPauseBtn.textContent = '⏸️';
    }
}

// Determine the category based on data content
function determineCategory(data) {
    if (!data) return 'generic';

    if (data.algorithm && typeof data.algorithm === 'string') {
        if (data.algorithm.includes('sort')) {
            return 'sorting';
        } else if (data.algorithm.includes('search')) {
            return 'searching';
        }
    }

    if (data.graph || (frames.length > 0 && (frames[0].nodes || frames[0].edges || frames[0].visited))) {
        return 'graph';
    }

    return 'generic';
}

// Start animation loop
function animateFrames(renderFunction, speed) {
    // Stop any existing animation
    if (animationTimer) {
        clearInterval(animationTimer);
    }

    // Calculate delay in milliseconds (higher speed = lower delay)
    const delay = 1000 / speed;

    // Start animation loop
    animationTimer = setInterval(() => {
        if (animationPaused) return;

        if (currentFrame >= frames.length) {
            clearInterval(animationTimer);
            return;
        }

        renderCurrentFrame(renderFunction);
        currentFrame++;

        // Update progress bar
        updateProgressBar(currentFrame / frames.length * 100);
    }, delay);
}

// Render the current frame using the appropriate renderer
function renderCurrentFrame(renderFunction) {
    if (currentFrame < 0 || currentFrame >= frames.length) return;

    const frame = frames[currentFrame];

    if (!frame) {
        console.error('Invalid frame at index', currentFrame);
        return;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Render frame
    renderFunction(ctx, frame, canvas.width, canvas.height);

    // Update info text
    updateInfoText(frame.info || '');
}

// Render a sorting algorithm frame
function renderSortingFrame(ctx, frame, width, height) {
    if (!frame || !frame.state) {
        ctx.fillStyle = '#f5f7fa';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#666';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No frame data available', width / 2, height / 2);
        return;
    }

    const array = Array.isArray(frame.state) ? frame.state : [];

    // No data to display
    if (array.length === 0) {
        ctx.fillStyle = '#f5f7fa';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#666';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Empty array', width / 2, height / 2);
        return;
    }

    // Find max value safely
    const maxValue = Math.max(...array.map(v => isNaN(v) ? 0 : v), 1);

    const barWidth = width / array.length;
    const barSpacing = Math.min(barWidth * 0.2, 4); // Space between bars, max 4px

    // Draw bars
    for (let i = 0; i < array.length; i++) {
        const value = array[i];
        const barHeight = (value / maxValue) * (height - 80); // Leave space for labels

        // Determine bar color (highlight current elements being processed)
        let color = '#4a6ee0'; // Default blue
        if (frame.highlight && frame.highlight.includes(i)) {
            color = '#e04a6e'; // Highlight color (red)
        }

        // Draw bar
        ctx.fillStyle = color;
        ctx.fillRect(
            i * barWidth + barSpacing / 2,
            height - barHeight - 40, // 40px from bottom for labels
            barWidth - barSpacing,
            barHeight
        );

        // Draw value label if there's enough space
        if (barWidth > 20) {
            ctx.fillStyle = '#333';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(
                value.toString(),
                i * barWidth + barWidth / 2,
                height - barHeight - 45
            );
        }

        // Draw index label
        ctx.fillStyle = '#666';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
            i.toString(),
            i * barWidth + barWidth / 2,
            height - 20
        );
    }

    // Draw frame number
    ctx.fillStyle = '#333';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Frame: ${currentFrame + 1}/${frames.length}`, 10, 20);
}

// Render a searching algorithm frame
function renderSearchingFrame(ctx, frame, width, height) {
    if (!frame || !frame.state) {
        ctx.fillStyle = '#f5f7fa';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#666';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No frame data available', width / 2, height / 2);
        return;
    }

    const array = Array.isArray(frame.state) ? frame.state : [];

    // No data to display
    if (array.length === 0) {
        ctx.fillStyle = '#f5f7fa';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#666';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Empty array', width / 2, height / 2);
        return;
    }

    // Find max value safely
    const maxValue = Math.max(...array.map(v => isNaN(v) ? 0 : v), 1);

    const barWidth = width / array.length;
    const barSpacing = Math.min(barWidth * 0.2, 4);

    // Draw bars (similar to sorting visualization)
    for (let i = 0; i < array.length; i++) {
        const value = array[i];
        const barHeight = (value / maxValue) * (height - 80);

        // Determine bar color
        let color = '#4a6ee0'; // Default blue

        // Special colors for binary search
        if (frame.highlight && frame.highlight.includes(i)) {
            color = '#e04a6e'; // Current element being checked (red)
        } else if (frame.range) {
            // If we have range info (for binary search)
            if (i < frame.range.left || i > frame.range.right) {
                color = '#cccccc'; // Eliminated range (gray)
            }
        }

        // Draw bar
        ctx.fillStyle = color;
        ctx.fillRect(
            i * barWidth + barSpacing / 2,
            height - barHeight - 40,
            barWidth - barSpacing,
            barHeight
        );

        // Draw value label if there's enough space
        if (barWidth > 20) {
            ctx.fillStyle = '#333';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(
                value.toString(),
                i * barWidth + barWidth / 2,
                height - barHeight - 45
            );
        }

        // Draw index label
        ctx.fillStyle = '#666';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
            i.toString(),
            i * barWidth + barWidth / 2,
            height - 20
        );
    }

    // Draw target value
    if (frame.target !== undefined) {
        ctx.fillStyle = '#333';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Target: ${frame.target}`, 10, 40);
    }

    // Draw frame number
    ctx.fillStyle = '#333';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Frame: ${currentFrame + 1}/${frames.length}`, 10, 20);
}

// Render a generic frame (fallback)
function renderGenericFrame(ctx, frame, width, height) {
    if (frame.image) {
        // If we have a pre-rendered image, use it
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, width, height);
        };
        img.src = frame.image;
        return;
    }

    // Draw placeholder
    ctx.fillStyle = '#f5f7fa';
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = '#666';
    ctx.font = '16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(
        'No visualization data available for this algorithm.',
        width / 2,
        height / 2
    );
}

// Create animation control buttons
// Create animation control buttons
function createAnimationControls() {
    const controlsContainer = document.createElement('div');
    controlsContainer.className = 'animation-controls';

    const controlsHTML = `
        <button class="restart" title="Restart">⏮️</button>
        <button class="step-backward" title="Previous Step">⏪</button>
        <button class="play-pause" title="Play/Pause">⏸️</button>
        <button class="step-forward" title="Next Step">⏩</button>
        <div class="progress-bar">
            <div class="progress-bar-fill"></div>
        </div>
    `;

    controlsContainer.innerHTML = controlsHTML;
    return controlsContainer;
}

// Set up control buttons event listeners
function setupControlListeners(renderFunction, speed) {
    const controls = document.querySelector('.animation-controls');
    if (!controls) return;

    // Remove existing listeners first to avoid duplicates
    const clone = controls.cloneNode(true);
    controls.parentNode.replaceChild(clone, controls);

    // Get new references
    const playPauseBtn = clone.querySelector('.play-pause');
    const stepForwardBtn = clone.querySelector('.step-forward');
    const stepBackwardBtn = clone.querySelector('.step-backward');
    const restartBtn = clone.querySelector('.restart');
    const progressBar = clone.querySelector('.progress-bar');

    // Play/Pause
    playPauseBtn.addEventListener('click', () => {
        animationPaused = !animationPaused;
        playPauseBtn.textContent = animationPaused ? '▶️' : '⏸️';

        if (!animationTimer && !animationPaused) {
            // Restart animation if it was stopped
            animateFrames(renderFunction, speed);
        }
    });

    // Step Forward
    stepForwardBtn.addEventListener('click', () => {
        if (animationTimer) {
            clearInterval(animationTimer);
            animationTimer = null;
        }

        animationPaused = true;
        playPauseBtn.textContent = '▶️';

        if (currentFrame < frames.length) {
            renderCurrentFrame(renderFunction);
            currentFrame++;
            updateProgressBar(currentFrame / frames.length * 100);
        }
    });

    // Step Backward
    stepBackwardBtn.addEventListener('click', () => {
        if (animationTimer) {
            clearInterval(animationTimer);
            animationTimer = null;
        }

        animationPaused = true;
        playPauseBtn.textContent = '▶️';

        if (currentFrame > 0) {
            currentFrame--;
            renderCurrentFrame(renderFunction);
            updateProgressBar(currentFrame / frames.length * 100);
        }
    });

    // Restart
    restartBtn.addEventListener('click', () => {
        if (animationTimer) {
            clearInterval(animationTimer);
            animationTimer = null;
        }

        currentFrame = 0;
        animationPaused = true;
        playPauseBtn.textContent = '▶️';

        renderCurrentFrame(renderFunction);
        updateProgressBar(0);
    });

    // Click on progress bar to jump to specific frame
    if (progressBar) {
        progressBar.addEventListener('click', (e) => {
            const rect = progressBar.getBoundingClientRect();
            const clickPosition = e.clientX - rect.left;
            const percentage = clickPosition / rect.width;

            // Calculate frame index based on percentage
            const frameIndex = Math.floor(percentage * frames.length);

            // Update current frame and render
            currentFrame = Math.max(0, Math.min(frameIndex, frames.length - 1));
            renderCurrentFrame(renderFunction);
            updateProgressBar(currentFrame / frames.length * 100);

            // Pause animation
            animationPaused = true;
            playPauseBtn.textContent = '▶️';

            if (animationTimer) {
                clearInterval(animationTimer);
                animationTimer = null;
            }
        });
    }
}

// Update progress bar
function updateProgressBar(percentage) {
    const progressBar = document.querySelector('.progress-bar-fill');
    if (progressBar) {
        progressBar.style.width = `${percentage}%`;
    }
}

// Update info text
function updateInfoText(text) {
    const infoText = document.getElementById('info-text');
    if (infoText) {
        infoText.textContent = text || '';
    }
}

// Show a placeholder message when no visualization is available
function showPlaceholder(message) {
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#666';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(message, canvas.width / 2, canvas.height / 2);
    }

    const infoText = document.getElementById('info-text');
    if (infoText) {
        infoText.textContent = '';
    }
}