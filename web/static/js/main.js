// main.js - Entry point for the JavaScript application
import { initVisualization, renderVisualization, clearVisualization } from './visualisation.js';
import { setupUIControls, getInputArray, getAnimationSpeed, getAlgorithmOptions } from './ui-controls.js';
import { fetchCategories, fetchAlgorithms, runAlgorithm, compareAlgorithms, fetchAlgorithmCode } from './algorithms.js';
import GraphUI from './graph-ui.js';

// DOM elements
const categoryList = document.getElementById('category-list');
const categoryTitle = document.getElementById('category-title');
const algorithmSelect = document.getElementById('algorithm-select');
const runButton = document.getElementById('run-button');
const tabButtons = document.querySelectorAll('.tab-button');
const tabPanes = document.querySelectorAll('.tab-pane');
const applyCustomInputBtn = document.getElementById('apply-custom-input');

// State
let currentCategory = null;
let currentAlgorithm = null;
let visualizationData = null;
let isAnimationRunning = false;
let graphUI = null;

// Render the list of algorithm categories
function renderCategories(categories) {
    categoryList.innerHTML = '';

    categories.forEach(category => {
        const li = document.createElement('li');
        li.textContent = category.name;
        li.dataset.id = category.id;
        li.addEventListener('click', () => selectCategory(category));
        categoryList.appendChild(li);
    });
}

// Handle category selection
async function selectCategory(category) {
    // Update UI
    document.querySelectorAll('#category-list li').forEach(li => {
        li.classList.remove('active');
        if (li.dataset.id === category.id) {
            li.classList.add('active');
        }
    });

    categoryTitle.textContent = category.name;
    currentCategory = category.id;

    // Load algorithms for this category
    try {
        const algorithms = await fetchAlgorithms(category.id);
        renderAlgorithms(algorithms);

        // Special handling for graph category
        if (category.id === 'graph') {
            initGraphUI();
        } else {
            // Clean up graph UI if it exists
            if (graphUI) {
                const graphContainer = document.getElementById('graph-editor-container');
                if (graphContainer) {
                    graphContainer.innerHTML = '';
                    graphContainer.style.display = 'none';
                }
            }
        }
    } catch (error) {
        console.error(`Failed to load algorithms for ${category.id}:`, error);
        showError(`Failed to load algorithms for ${category.name}.`);
    }
}

// Initialize the graph editor UI
function initGraphUI() {
    // Create or show graph editor container
    let graphContainer = document.getElementById('graph-editor-container');

    if (!graphContainer) {
        graphContainer = document.createElement('div');
        graphContainer.id = 'graph-editor-container';
        graphContainer.className = 'graph-editor-container';

        // Insert after the controls panel
        const controlsPanel = document.querySelector('.controls-panel');
        controlsPanel.parentNode.insertBefore(graphContainer, controlsPanel.nextSibling);
    }

    graphContainer.style.display = 'block';

    // Initialize the graph UI
    graphUI = new GraphUI(graphContainer);
}

// Render the dropdown of algorithms
function renderAlgorithms(algorithms) {
    algorithmSelect.innerHTML = '';
    algorithmSelect.disabled = algorithms.length === 0;
    runButton.disabled = algorithms.length === 0;

    if (algorithms.length === 0) {
        const option = document.createElement('option');
        option.textContent = 'No algorithms available';
        algorithmSelect.appendChild(option);
        return;
    }

    algorithms.forEach(algorithm => {
        const option = document.createElement('option');
        option.value = algorithm.id;
        option.textContent = `${algorithm.name} - ${algorithm.complexity}`;
        algorithmSelect.appendChild(option);
    });

    // Select the first algorithm by default
    selectAlgorithm(algorithms[0]);
}

// Handle algorithm selection
function selectAlgorithm(algorithm) {
    currentAlgorithm = algorithm.id;

    // Clear any existing visualization
    clearVisualization();

    // Update the code tab with this algorithm's implementation
    loadAlgorithmCode(currentCategory, currentAlgorithm);

    // Update other tabs as needed
    loadAlgorithmPerformance(currentCategory, currentAlgorithm);
    loadAlgorithmTheory(currentCategory, currentAlgorithm);
    loadAlgorithmApplications(currentCategory, currentAlgorithm);
}

// Load and display the algorithm's code
function loadAlgorithmCode(category, algorithmId) {
    const codeElement = document.getElementById('algorithm-code');
    codeElement.textContent = 'Loading code...';

    fetchAlgorithmCode(category, algorithmId)
        .then(data => {
            codeElement.textContent = data.code || 'Code not available';
        })
        .catch(error => {
            console.error('Failed to load code:', error);
            codeElement.textContent = 'Failed to load code';
        });
}

// Load and display performance data
function loadAlgorithmPerformance(category, algorithmId) {
    const performanceChart = document.getElementById('performance-chart');
    const performanceExplanation = document.getElementById('performance-explanation');

    performanceChart.innerHTML = 'Loading performance data...';
    performanceExplanation.innerHTML = '';

    // Fetch performance data based on category and algorithm
    fetch(`/api/${category}/performance?algorithm=${algorithmId}`)
        .then(response => response.json())
        .then(data => {
            if (data.chart) {
                // If we have a chart image
                performanceChart.innerHTML = `<img src="${data.chart}" alt="Performance Chart" style="max-width:100%;">`;
            } else {
                performanceChart.innerHTML = 'No performance data available';
            }

            if (data.explanation) {
                performanceExplanation.innerHTML = data.explanation;
            }
        })
        .catch(error => {
            console.error('Failed to load performance data:', error);
            performanceChart.innerHTML = 'Failed to load performance data';
        });
}

// Load theoretical background
function loadAlgorithmTheory(category, algorithmId) {
    const theoryContent = document.getElementById('theory-content');
    theoryContent.innerHTML = 'Loading theoretical background...';

    fetch(`/api/${category}/theory?algorithm=${algorithmId}`)
        .then(response => response.json())
        .then(data => {
            if (data.content) {
                theoryContent.innerHTML = data.content;
            } else {
                theoryContent.innerHTML = `
                    <h3>${algorithmId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
                    <p>Theoretical background information is not available for this algorithm.</p>
                `;
            }
        })
        .catch(error => {
            console.error('Failed to load theory content:', error);
            theoryContent.innerHTML = 'Failed to load theoretical background';
        });
}

// Load practical applications
function loadAlgorithmApplications(category, algorithmId) {
    const applicationsContent = document.getElementById('applications-content');
    applicationsContent.innerHTML = 'Loading practical applications...';

    fetch(`/api/${category}/applications?algorithm=${algorithmId}`)
        .then(response => response.json())
        .then(data => {
            if (data.content) {
                applicationsContent.innerHTML = data.content;
            } else {
                applicationsContent.innerHTML = `
                    <h3>Applications of ${algorithmId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
                    <p>Application information is not available for this algorithm.</p>
                `;
            }
        })
        .catch(error => {
            console.error('Failed to load applications content:', error);
            applicationsContent.innerHTML = 'Failed to load practical applications';
        });
}

// Setup all event listeners
function setupEventListeners() {
    // Run button
    runButton.addEventListener('click', runCurrentAlgorithm);

    // Algorithm select dropdown
    algorithmSelect.addEventListener('change', () => {
        const selectedId = algorithmSelect.value;
        const algorithms = Array.from(algorithmSelect.options).map(option => ({
            id: option.value,
            name: option.textContent.split(' - ')[0]
        }));

        const selected = algorithms.find(algo => algo.id === selectedId);
        if (selected) {
            selectAlgorithm(selected);
        }
    });

    // Tab switching
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;

            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Show active tab content
            tabPanes.forEach(pane => pane.classList.remove('active'));
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });

    // Custom input button
    applyCustomInputBtn.addEventListener('click', () => {
        const inputField = document.getElementById('custom-input-field');
        // Implementation handled by ui-controls.js
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
        // Prevent shortcuts when typing in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        switch (e.key) {
            case 'r':
            case 'R':
                if (!runButton.disabled) {
                    runCurrentAlgorithm();
                }
                break;
        }
    });
}

// Show error message
function showError(message) {
    const errorContainer = document.getElementById('error-container') || createErrorContainer();

    const errorMessage = document.createElement('div');
    errorMessage.className = 'error-message';
    errorMessage.innerHTML = `
        <span>${message}</span>
        <button class="close-btn">×</button>
    `;

    errorContainer.appendChild(errorMessage);

    // Add close button handler
    errorMessage.querySelector('.close-btn').addEventListener('click', () => {
        errorContainer.removeChild(errorMessage);
    });

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorContainer.contains(errorMessage)) {
            errorContainer.removeChild(errorMessage);
        }
    }, 5000);
}

// Create error container if it doesn't exist
function createErrorContainer() {
    const container = document.createElement('div');
    container.id = 'error-container';
    container.className = 'error-container';
    document.body.appendChild(container);
    return container;
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);


// main.js - Updated to expose runCurrentAlgorithm for the auto-update feature

// Initialize the application
async function initApp() {
    try {
        // Setup UI controls
        setupUIControls();

        // Initialize visualization container
        initVisualization();

        // Load categories
        const categories = await fetchCategories();
        renderCategories(categories);

        // Setup event listeners
        setupEventListeners();

        // Make runCurrentAlgorithm accessible globally for auto-update
        window.runCurrentAlgorithm = runCurrentAlgorithm;
    } catch (error) {
        console.error('Failed to initialize app:', error);
        showError('Failed to load the application. Please try again later.');
    }
}

// This update fixes the issue in main.js where the input array wasn't being correctly passed to the API
async function runCurrentAlgorithm() {
    if (!currentCategory || !currentAlgorithm) return;

    // Disable the run button while running
    runButton.disabled = true;
    runButton.textContent = 'Running...';

    try {
        let options = getAlgorithmOptions(currentCategory, currentAlgorithm);

        // Special handling for graph algorithms
        if (currentCategory === 'graph' && graphUI) {
            const graphData = graphUI.getGraphData();
            const algorithmParams = graphUI.getAlgorithmParams();

            options = {
                ...options,
                graph: graphData,
                source: algorithmParams.startVertex,
                target: algorithmParams.targetVertex
            };
        } else {
            // For non-graph algorithms, use the input array
            const inputArray = getInputArray();

            // Fix the API call to correctly separate input from other options
            visualizationData = await runAlgorithm(currentCategory, currentAlgorithm, inputArray, options);
            renderVisualization(visualizationData, getAnimationSpeed());

            runButton.disabled = false;
            runButton.textContent = 'Run';
            return;
        }

        // This is for graph algorithms only
        visualizationData = await runAlgorithm(currentCategory, currentAlgorithm, null, options);
        renderVisualization(visualizationData, getAnimationSpeed());
    } catch (error) {
        console.error('Failed to run algorithm:', error);
        showError('Failed to run the algorithm. Please try again.');
    } finally {
        runButton.disabled = false;
        runButton.textContent = 'Run';
    }
}