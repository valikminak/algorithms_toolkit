// visualization.js - Handles all visualization rendering

let visualizationContainer;
let animationTimer;
let currentFrame = 0;
let frames = [];

// Initialize the visualization container
export function initVisualization() {
    visualizationContainer = document.getElementById('visualization-container');
}

// Render visualization data
export function renderVisualization(data, speed) {
    clearVisualization();

    if (!data || !data.visualization) {
        showPlaceholder('No visualization data available');
        return;
    }

    frames = data.visualization;

    // Determine which type of visualization to use based on the data
    if (isArrayVisualization(data)) {
        renderArrayVisualization(frames, speed);
    } else if (isGraphVisualization(data)) {
        renderGraphVisualization(frames, speed);
    } else if (isTreeVisualization(data)) {
        renderTreeVisualization(frames, speed);
    } else {
        // Generic visualization - just show the frames as images
        renderImageFrames(frames, speed);
    }
}

// Clear the visualization and stop any animations
export function clearVisualization() {
    if (animationTimer) {
        clearInterval(animationTimer);
        animationTimer = null;
    }

    visualizationContainer.innerHTML = '';
    currentFrame = 0;
    frames = [];
}

// Check if we should use array-based visualization (like for sorting)
function isArrayVisualization(data) {
    return data.category === 'sorting' ||
           (frames.length > 0 && Array.isArray(frames[0].state));
}

// Check if we should use graph-based visualization
function isGraphVisualization(data) {
    return data.category === 'graph' ||
           (frames.length > 0 && frames[0].nodes && frames[0].edges);
}

// Check if we should use tree-based visualization
function isTreeVisualization(data) {
    return data.category === 'tree' ||
           (frames.length > 0 && frames[0].root);
}

// Render array-based visualization (like for sorting algorithms)
function renderArrayVisualization(frames, speed) {
    const maxValue = getMaxValue(frames);
    const barContainer = document.createElement('div');
    barContainer.className = 'bar-container';
    visualizationContainer.appendChild(barContainer);

    // Create info text element
    const infoText = document.createElement('div');
    infoText.className = 'info-text';
    visualizationContainer.appendChild(infoText);

    // Create animation controls
    const controls = createAnimationControls();
    visualizationContainer.appendChild(controls);

    // Start animation
    animateFrames();

    function animateFrames() {
        // Stop any existing animation
        if (animationTimer) {
            clearInterval(animationTimer);
        }

        // Calculate speed in milliseconds (invert so higher = faster)
        const interval = 1000 / speed;

        // Start animation
        animationTimer = setInterval(() => {
            if (currentFrame >= frames.length) {
                clearInterval(animationTimer);
                return;
            }

            const frame = frames[currentFrame];
            renderArrayFrame(frame, barContainer, maxValue);
            infoText.textContent = frame.info || '';
            currentFrame++;

            // Update progress bar in controls
            updateProgressBar(currentFrame / frames.length * 100);

        }, interval);
    }

    // Event handlers for animation controls
    controls.querySelector('.play-pause').addEventListener('click', togglePlayPause);
    controls.querySelector('.step-forward').addEventListener('click', stepForward);
    controls.querySelector('.step-backward').addEventListener('click', stepBackward);
    controls.querySelector('.restart').addEventListener('click', restart);

    function togglePlayPause() {
        const button = controls.querySelector('.play-pause');
        if (animationTimer) {
            clearInterval(animationTimer);
            animationTimer = null;
            button.textContent = '▶️';
        } else {
            button.textContent = '⏸️';
            animateFrames();
        }
    }

    function stepForward() {
        if (animationTimer) {
            clearInterval(animationTimer);
            animationTimer = null;
            controls.querySelector('.play-pause').textContent = '▶️';
        }

        if (currentFrame < frames.length) {
            const frame = frames[currentFrame];
            renderArrayFrame(frame, barContainer, maxValue);
            infoText.textContent = frame.info || '';
            currentFrame++;
            updateProgressBar(currentFrame / frames.length * 100);
        }
    }

    function stepBackward() {
        if (animationTimer) {
            clearInterval(animationTimer);
            animationTimer = null;
            controls.querySelector('.play-pause').textContent = '▶️';
        }

        if (currentFrame > 0) {
            currentFrame -= 2;
            if (currentFrame < 0) currentFrame = 0;

            const frame = frames[currentFrame];
            renderArrayFrame(frame, barContainer, maxValue);
            infoText.textContent = frame.info || '';
            currentFrame++;
            updateProgressBar(currentFrame / frames.length * 100);
        }
    }

    function restart() {
        if (animationTimer) {
            clearInterval(animationTimer);
            animationTimer = null;
            controls.querySelector('.play-pause').textContent = '▶️';
        }

        currentFrame = 0;
        const frame = frames[currentFrame];
        renderArrayFrame(frame, barContainer, maxValue);
        infoText.textContent = frame.info || '';
        currentFrame++;
        updateProgressBar(currentFrame / frames.length * 100);
    }

    function updateProgressBar(percentage) {
        const progressBar = controls.querySelector('.progress-bar-fill');
        progressBar.style.width = `${percentage}%`;
    }
}

// Render a single frame of array visualization
function renderArrayFrame(frame, container, maxValue) {
    const array = frame.state || [];

    // Clear the container
    container.innerHTML = '';

    // Create bars for each element
    array.forEach((value, index) => {
        const bar = document.createElement('div');
        bar.className = 'bar';

        // Calculate height as percentage of max value
        const heightPercent = (value / maxValue) * 100;
        bar.style.height = `${heightPercent}%`;

        // Highlight specific elements if needed
        if (frame.highlight && frame.highlight.includes(index)) {
            bar.classList.add('highlight');
        }

        // Add value label
        const label = document.createElement('span');
        label.className = 'bar-label';
        label.textContent = value;
        bar.appendChild(label);

        container.appendChild(bar);
    });
}

// Render image-based frames (for more complex visualizations)
function renderImageFrames(frames, speed) {
    const imageContainer = document.createElement('div');
    imageContainer.className = 'image-container';
    visualizationContainer.appendChild(imageContainer);

    // Create info text element
    const infoText = document.createElement('div');
    infoText.className = 'info-text';
    visualizationContainer.appendChild(infoText);

    // Create animation controls
    const controls = createAnimationControls();
    visualizationContainer.appendChild(controls);

    // Start animation
    animateFrames();

    function animateFrames() {
        // Implementation similar to renderArrayVisualization
        // but for image frames
    }

    // Event handlers for controls would be similar to renderArrayVisualization
}

// Create animation control buttons
function createAnimationControls() {
    const controlsContainer = document.createElement('div');
    controlsContainer.className = 'animation-controls';

    const controlsHTML = `
        <button class="restart">⏮️</button>
        <button class="step-backward">⏪</button>
        <button class="play-pause">⏸️</button>
        <button class="step-forward">⏩</button>
        <div class="progress-bar">
            <div class="progress-bar-fill"></div>
        </div>
    `;

    controlsContainer.innerHTML = controlsHTML;
    return controlsContainer;
}

// Get the maximum value from all frames (for scaling visualizations)
function getMaxValue(frames) {
    let max = 0;

    frames.forEach(frame => {
        if (frame.state) {
            const frameMax = Math.max(...frame.state);
            max = Math.max(max, frameMax);
        }
    });

    return max;
}

// Show a placeholder message when no visualization is available
function showPlaceholder(message) {
    visualizationContainer.innerHTML = `
        <div class="placeholder-message">
            ${message}
        </div>
    `;
}

// Rendering functions for other visualization types would be added here
function renderGraphVisualization(frames, speed) {
    // Implementation for graph visualization
}

function renderTreeVisualization(frames, speed) {
    // Implementation for tree visualization
}