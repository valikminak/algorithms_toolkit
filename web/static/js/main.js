// main.js - Entry point for the JavaScript application
import { initVisualization, renderVisualization, clearVisualization } from './visualization.js';
import { setupUIControls, getInputArray, getAnimationSpeed } from './ui-controls.js';
import { fetchCategories, fetchAlgorithms, runAlgorithm, compareAlgorithms } from './algorithms.js';

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
    } catch (error) {
        console.error(`Failed to load algorithms for ${category.id}:`, error);
        showError(`Failed to load algorithms for ${category.name}.`);
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

// Run the selected algorithm
async function runCurrentAlgorithm() {
    if (!currentCategory || !currentAlgorithm) return;

    runButton.disabled = true;
    runButton.textContent = 'Running...';

    try {
        const inputArray = getInputArray();
        visualizationData = await runAlgorithm(currentCategory, currentAlgorithm, inputArray);
        renderVisualization(visualizationData, getAnimationSpeed());
    } catch (error) {
        console.error('Failed to run algorithm:', error);
        showError('Failed to run the algorithm. Please try again.');
    } finally {
        runButton.disabled = false;
        runButton.textContent = 'Run';
    }
}

// Load and display the algorithm's code
function loadAlgorithmCode(category, algorithmId) {
    const codeElement = document.getElementById('algorithm-code');

    // In a real implementation, you would fetch this from the server
    // For now, just show a placeholder
    codeElement.textContent = `// Loading code for ${algorithmId}...`;

    fetch(`/api/${category}/code?algorithm=${algorithmId}`)
        .then(response => response.json())
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

    // In a real implementation, you would fetch performance data
    // For now, just show placeholders

    // This would be an async call in a real implementation
    // that would load and render a chart
}

// Load theoretical background
function loadAlgorithmTheory(category, algorithmId) {
    const theoryContent = document.getElementById('theory-content');
    theoryContent.innerHTML = 'Loading theoretical background...';

    // In a real implementation, you would fetch theory content
}

// Load practical applications
function loadAlgorithmApplications(category, algorithmId) {
    const applicationsContent = document.getElementById('applications-content');
    applicationsContent.innerHTML = 'Loading practical applications...';

    // In a real implementation, you would fetch applications content
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
}

// Show error message
function showError(message) {
    // You could create a toast notification system here
    alert(message);
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);