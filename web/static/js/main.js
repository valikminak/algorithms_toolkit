// main.js - Entry point for the JavaScript application
import { fetchCategories, fetchAlgorithms, runAlgorithm, compareAlgorithms, fetchAlgorithmCode } from './common/algorithms.js';
import { setupUIControls, getInputArray, getAnimationSpeed, getAlgorithmOptions } from './common/ui-controls.js';

// DOM elements
const categoryList = document.getElementById('category-list');
const categoryTitle = document.getElementById('category-title');
const algorithmSelect = document.getElementById('algorithm-select');
const runButton = document.getElementById('run-button');
const tabButtons = document.querySelectorAll('.tab-button');
const tabPanes = document.querySelectorAll('.tab-pane');
const applyCustomInputBtn = document.getElementById('apply-custom-input');
const visualizationContainer = document.getElementById('visualization-container');

// State
let currentCategory = null;
let currentAlgorithm = null;
let visualizationData = null;
let currentVisualizer = null;

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);

// Initialize the application
async function initApp() {
    try {
        // Setup UI controls
        setupUIControls();

        // Load categories
        const categories = await fetchCategories();
        renderCategories(categories);

        // Setup event listeners
        setupEventListeners();
    } catch (error) {
        console.error('Failed to initialize app:', error);
        showError('Failed to load the application. Please try again later.');
    }
}

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
        const graphContainer = document.getElementById('graph-editor-container');
        if (category.id === 'graph') {
            initGraphVisualizer();
            if (graphContainer) {
                graphContainer.style.display = 'block';
            }
        } else {
            // Hide graph UI if not on graph category
            if (graphContainer) {
                graphContainer.style.display = 'none';
            }
        }

        // Clear current visualization
        clearVisualization();
    } catch (error) {
        console.error(`Failed to load algorithms for ${category.id}:`, error);
        showError(`Failed to load algorithms for ${category.name}.`);
    }
}

// Initialize the graph visualizer
function initGraphVisualizer() {
    try {
        // Check if graph visualization modules are available
        if (window.GraphUI && window.GraphVisualizer) {
            // Create graph UI if it doesn't exist
            const graphContainer = document.getElementById('graph-editor-container');
            if (graphContainer && !graphContainer.querySelector('.graph-editor')) {
                window.graphUI = new GraphUI(graphContainer);
            }
        } else {
            console.warn('Graph visualization modules not loaded');
        }
    } catch (error) {
        console.error('Failed to initialize graph visualizer:', error);
    }
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
    currentAlgorithm = algorithms[0].id;
    loadAlgorithmDetails(currentCategory, algorithms[0].id);
}

// Load details for the selected algorithm
function loadAlgorithmDetails(category, algorithmId) {
    // Load code implementation
    loadAlgorithmCode(category, algorithmId);

    // Load other tabs as needed
    loadAlgorithmPerformance(category, algorithmId);
    loadAlgorithmTheory(category, algorithmId);
    loadAlgorithmApplications(category, algorithmId);
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
        currentAlgorithm = selectedId;
        loadAlgorithmDetails(currentCategory, selectedId);
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

// Run the current algorithm
async function runCurrentAlgorithm() {
    if (!currentCategory || !currentAlgorithm) return;

    // Disable the run button while running
    runButton.disabled = true;
    runButton.textContent = 'Running...';

    try {
        let options = getAlgorithmOptions(currentCategory, currentAlgorithm);

        // Special handling for graph algorithms
        if (currentCategory === 'graph' && window.graphUI) {
            const graphData = window.graphUI.getGraphData();
            const algorithmParams = window.graphUI.getAlgorithmParams();

            options = {
                ...options,
                graph: graphData,
                source: algorithmParams.startVertex,
                target: algorithmParams.targetVertex
            };
        } else {
            // For non-graph algorithms, use the input array
            const inputArray = getInputArray();
            options.input = inputArray;
        }

        // Run the algorithm
        visualizationData = await runAlgorithm(currentCategory, currentAlgorithm, null, options);

        // Visualize the result
        visualizeResult(visualizationData);
    } catch (error) {
        console.error('Failed to run algorithm:', error);
        showError('Failed to run the algorithm. Please try again.');
    } finally {
        runButton.disabled = false;
        runButton.textContent = 'Run';
    }
}

// Visualize algorithm result using the appropriate visualizer
function visualizeResult(data) {
    if (!data || !data.visualization) {
        showError('No visualization data available');
        return;
    }

    // Clear previous visualization
    clearVisualization();

    // Determine which visualizer to use based on category
    const category = data.category || currentCategory;

    try {
        switch (category) {
            case 'sorting':
                if (window.SortingVisualizer) {
                    const frames = window.SortingVisualizer.prepareVisualizationData(data);

                    // Initialize visualizer if needed
                    if (!currentVisualizer || !(currentVisualizer instanceof window.SortingVisualizer)) {
                        currentVisualizer = new window.SortingVisualizer('visualization-container', {
                            animationSpeed: getAnimationSpeed()
                        });
                    } else {
                        currentVisualizer.setAnimationSpeed(getAnimationSpeed());
                    }

                    // Set data and start animation
                    currentVisualizer.setData(frames);
                    currentVisualizer.play();
                } else {
                    fallbackVisualization(data);
                }
                break;

            case 'searching':
                if (window.SearchingVisualizer) {
                    const frames = window.SearchingVisualizer.prepareVisualizationData(data);

                    // Initialize visualizer if needed
                    if (!currentVisualizer || !(currentVisualizer instanceof window.SearchingVisualizer)) {
                        currentVisualizer = new window.SearchingVisualizer('visualization-container', {
                            animationSpeed: getAnimationSpeed()
                        });
                    } else {
                        currentVisualizer.setAnimationSpeed(getAnimationSpeed());
                    }

                    // Set data and start animation
                    currentVisualizer.setData(frames);
                    currentVisualizer.play();
                } else {
                    fallbackVisualization(data);
                }
                break;

            case 'graph':
                if (window.GraphVisualizer) {
                    // Use the specialized graph visualizer
                    if (!currentVisualizer || !(currentVisualizer instanceof window.GraphVisualizer)) {
                        currentVisualizer = new window.GraphVisualizer('visualization-container');
                    }

                    currentVisualizer.initialize(data);
                    currentVisualizer.setSpeed(getAnimationSpeed());

                    // Attach controls
                    const controls = {
                        playPause: document.querySelector('.play-pause'),
                        next: document.querySelector('.step-forward'),
                        prev: document.querySelector('.step-backward'),
                        restart: document.querySelector('.restart'),
                        progressBar: document.querySelector('.progress-bar'),
                        progressIndicator: document.querySelector('.progress-bar-fill')
                    };

                    currentVisualizer.attachControls(controls);
                    currentVisualizer.startAnimation(getAnimationSpeed());
                } else {
                    fallbackVisualization(data);
                }
                break;

            default:
                // Use base visualizer for other types
                if (window.VisualizerBase) {
                    if (!currentVisualizer || !(currentVisualizer instanceof window.VisualizerBase)) {
                        currentVisualizer = new window.VisualizerBase('visualization-container', {
                            animationSpeed: getAnimationSpeed()
                        });
                    }

                    currentVisualizer.setData(data.visualization);
                    currentVisualizer.play();
                } else {
                    fallbackVisualization(data);
                }
        }
    } catch (error) {
        console.error('Visualization error:', error);
        showError('Failed to visualize the algorithm result');
        fallbackVisualization(data);
    }
}

// Fallback visualization when specialized visualizers are not available
function fallbackVisualization(data) {
    if (!data || !data.visualization) {
        visualizationContainer.innerHTML = '<div class="placeholder-message">No visualization data available</div>';
        return;
    }

    // Simple placeholder visualization
    visualizationContainer.innerHTML = `
        <div class="placeholder-message">
            <p>Visualization available but specialized visualizer not loaded.</p>
            <p>Algorithm: ${data.algorithm}</p>
            <p>Category: ${data.category}</p>
            <p>Execution time: ${data.execution_time ? data.execution_time.toFixed(6) + 's' : 'Unknown'}</p>
        </div>
    `;
}

// Clear the current visualization
function clearVisualization() {
    if (currentVisualizer) {
        currentVisualizer.reset();
    } else {
        visualizationContainer.innerHTML = '';
    }
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

// Make runCurrentAlgorithm available globally for auto-update feature
window.runCurrentAlgorithm = runCurrentAlgorithm;