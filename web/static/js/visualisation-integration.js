// Import specialized visualizers
import {initStringVisualization, renderStringVisualization} from './string-visualisation.js';
import {initDPVisualization, renderDPVisualization} from './dp-visualisation.js';
import {initNetworkFlowVisualization, renderNetworkFlowVisualization} from './network-flow-visualisation.js';

// Add these imports to the top of your existing visualization.js file
import GraphVisualizer from './graph-visualisation.js';

// Keep track of active visualizer
let activeVisualizer = null;
let activeVisualizerType = null;
let animationTimer = null;
let currentFrame = 0;
let frames = [];
let animationPaused = false;
let graphVisualizer = null;
let stringVisualizer = null;
let dpVisualizer = null;
let flowVisualizer = null;


/**
 * Enhanced version of the renderVisualization function that supports
 * all specialized visualizers
 *
 * @param {Object} data - The visualization data
 * @param {number} speed - Animation speed (1-10)
 */
export function renderVisualization(data, speed) {
    clearVisualization();

    if (!data || !data.visualization && !data.steps) {
        showPlaceholder('No visualization data available');
        return;
    }

    // Determine visualization type based on category or algorithm name
    const category = data.category || determineCategory(data);

    // Select appropriate visualizer based on category
    switch (category) {
        case 'string':
            activeVisualizerType = 'string';
            renderStringVisualization(data, speed);
            break;

        case 'dp':
        case 'dynamic_programming':
            activeVisualizerType = 'dp';
            renderDPVisualization(data, speed);
            break;

        case 'flow':
        case 'network_flow':
            activeVisualizerType = 'flow';
            renderNetworkFlowVisualization(data, speed);
            break;

        case 'graph':
            activeVisualizerType = 'graph';
            renderGraphVisualization(data, speed);
            break;

        case 'sorting':
            activeVisualizerType = 'sorting';
            renderSortingVisualization(data, speed);
            break;

        case 'searching':
            activeVisualizerType = 'searching';
            renderSearchingVisualization(data, speed);
            break;

        default:
            activeVisualizerType = 'generic';
            renderGenericVisualization(data, speed);
    }

    // Set up controls for the visualization
    setupControlListeners();
}

/**
 * Improved version of clearVisualization that handles all visualizer types
 */
// These would be module-level variables in visualization.js


/**
 * Clear the visualization and stop any animations
 */
export function clearVisualization() {
    // Stop any running animation
    if (animationTimer) {
        clearInterval(animationTimer);
        animationTimer = null;
    }

    // Clear canvas
    const canvas = document.getElementById('visualization-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    // Reset specific visualizer if active
    switch (activeVisualizerType) {
        case 'string':
            if (stringVisualizer) {
                stringVisualizer.reset();
            }
            break;

        case 'dp':
            if (dpVisualizer) {
                dpVisualizer.reset();
            }
            break;

        case 'flow':
            if (flowVisualizer) {
                flowVisualizer.reset();
            }
            break;

        case 'graph':
            if (graphVisualizer) {
                graphVisualizer.reset();
            }
            break;
    }

    // Reset info text
    const infoText = document.getElementById('info-text');
    if (infoText) {
        infoText.textContent = '';
    }

    // Reset progress bar
    const progressBar = document.querySelector('.progress-bar-fill');
    if (progressBar) {
        progressBar.style.width = '0%';
    }

    // Reset play/pause button
    const playPauseBtn = document.querySelector('.play-pause');
    if (playPauseBtn) {
        playPauseBtn.textContent = '▶️';
    }

    // Reset state variables
    currentFrame = 0;
    frames = [];
    animationPaused = false;
    activeVisualizerType = null;
}

/**
 * Determine the category based on data content
 * @param {Object} data - The visualization data
 * @returns {string} - The detected category
 */
function determineCategory(data) {
    // First check for explicit category
    if (data.category) return data.category;

    // Check the algorithm name
    if (data.algorithm) {
        const algoName = data.algorithm.toLowerCase();

        if (algoName.includes('kmp') || algoName.includes('rabin_karp') ||
            algoName.includes('boyer') || algoName.includes('pattern') ||
            algoName.includes('string')) {
            return 'string';
        }

        if (algoName.includes('knapsack') || algoName.includes('lcs') ||
            algoName.includes('chain') || algoName.includes('dp')) {
            return 'dp';
        }

        if (algoName.includes('flow') || algoName.includes('ford') ||
            algoName.includes('karp') || algoName.includes('cut')) {
            return 'flow';
        }

        if (algoName.includes('sort')) {
            return 'sorting';
        }

        if (algoName.includes('search')) {
            return 'searching';
        }

        if (algoName.includes('graph') || algoName.includes('path') ||
            algoName.includes('bfs') || algoName.includes('dfs')) {
            return 'graph';
        }
    }

    // Check for data structure indicators
    if (data.frames && data.frames.length > 0) {
        const frame = data.frames[0];

        if (frame.pattern || frame.text) {
            return 'string';
        }

        if (frame.dp_table) {
            return 'dp';
        }

        if (frame.residual_graph || frame.flow_graph) {
            return 'flow';
        }

        if (frame.state && Array.isArray(frame.state)) {
            return 'sorting';
        }

        if (frame.nodes || frame.edges || frame.visited) {
            return 'graph';
        }

        if (frame.target !== undefined) {
            return 'searching';
        }
    }

    // Default to generic
    return 'generic';
}

/**
 * Set up event listeners for control buttons
 */
function setupControlListeners() {
    const controls = document.querySelector('.animation-controls');
    if (!controls) return;

    // Remove existing listeners
    const clone = controls.cloneNode(true);
    controls.parentNode.replaceChild(clone, controls);

    // Get new references
    const playPauseBtn = clone.querySelector('.play-pause');
    const stepForwardBtn = clone.querySelector('.step-forward');
    const stepBackwardBtn = clone.querySelector('.step-backward');
    const restartBtn = clone.querySelector('.restart');
    const progressBar = clone.querySelector('.progress-bar');

    // Depending on active visualizer type, attach the appropriate handlers
    // This is handled by each individual visualizer's attachControls method
}

/**
 * Initialize all visualization components
 */
export function initVisualization() {
    const visualizationContainer = document.getElementById('visualization-container');
    if (!visualizationContainer) return;

    // Create canvas if it doesn't exist
    let canvas = document.getElementById('visualization-canvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'visualization-canvas';
        canvas.className = 'visualization-canvas';
        canvas.width = visualizationContainer.clientWidth || 800;
        canvas.height = 400;
        visualizationContainer.appendChild(canvas);
    }

    // Create info text element
    let infoText = document.getElementById('info-text');
    if (!infoText) {
        infoText = document.createElement('div');
        infoText.className = 'info-text';
        infoText.id = 'info-text';
        visualizationContainer.appendChild(infoText);
    }

    // Add animation controls
    let controlsContainer = document.querySelector('.animation-controls');
    if (!controlsContainer) {
        controlsContainer = document.createElement('div');
        controlsContainer.className = 'animation-controls';

        controlsContainer.innerHTML = `
            <button class="restart" title="Restart">⏮️</button>
            <button class="step-backward" title="Previous Step">⏪</button>
            <button class="play-pause" title="Play/Pause">⏸️</button>
            <button class="step-forward" title="Next Step">⏩</button>
            <div class="progress-bar">
                <div class="progress-bar-fill"></div>
            </div>
        `;

        visualizationContainer.appendChild(controlsContainer);
    }

    // Initialize the placeholder
    showPlaceholder('Select an algorithm to visualize');
}

/**
 * Show a placeholder message when no visualization is available
 */
function showPlaceholder(message) {
    const visualizationContainer = document.getElementById('visualization-container');
    if (!visualizationContainer) return;

    // Clear container
    visualizationContainer.innerHTML = '';

    // Create placeholder message
    const placeholder = document.createElement('div');
    placeholder.className = 'placeholder-message';
    placeholder.textContent = message;

    visualizationContainer.appendChild(placeholder);
}

// Export the main visualization functions
export default {
    initVisualization,
    renderVisualization,
    clearVisualization
};