const inputSizeSlider = document.getElementById('input-size');
const inputSizeValue = document.getElementById('input-size-value');
const animationSpeedSlider = document.getElementById('animation-speed');
const speedValue = document.getElementById('speed-value');
const customInputField = document.getElementById('custom-input-field');
const applyCustomInputBtn = document.getElementById('apply-custom-input');

// State
let currentInputArray = [];
let useCustomInput = false;
let autoUpdateEnabled = true; // Default to auto-update

/**
 * Initialize UI controls and set up event listeners
 */
export function setupUIControls() {
    // Initialize input array with random data
    generateRandomArray(parseInt(inputSizeSlider.value));

    // Create auto-update toggle
    createAutoUpdateToggle();

    // Set up event listeners
    inputSizeSlider.addEventListener('input', handleInputSizeChange);
    animationSpeedSlider.addEventListener('input', handleSpeedChange);
    applyCustomInputBtn.addEventListener('click', handleCustomInputApply);
    customInputField.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
            handleCustomInputApply();
        }
    });

    // Update visual indicators for initial values
    inputSizeValue.textContent = inputSizeSlider.value;
    speedValue.textContent = animationSpeedSlider.value;
}

/**
 * Add auto-update toggle to the control panel
 */
function createAutoUpdateToggle() {
    const controlGroup = document.createElement('div');
    controlGroup.className = 'control-group';

    const label = document.createElement('label');
    label.htmlFor = 'auto-update';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = 'auto-update';
    checkbox.checked = autoUpdateEnabled;
    checkbox.addEventListener('change', (e) => {
        autoUpdateEnabled = e.target.checked;
    });

    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(' Auto-update visualization'));

    controlGroup.appendChild(label);

    // Add to control panel (right after animation speed)
    const inputControls = document.querySelector('.input-controls');
    if (inputControls) {
        inputControls.appendChild(controlGroup);
    }
}

/**
 * Handle input size slider change
 */
function handleInputSizeChange() {
    const size = parseInt(inputSizeSlider.value);
    inputSizeValue.textContent = size;

    // Only regenerate array if not using custom input
    if (!useCustomInput) {
        generateRandomArray(size);

        // Automatically run algorithm if auto-update is enabled
        if (autoUpdateEnabled && window.runCurrentAlgorithm) {
            window.runCurrentAlgorithm();
        }
    }
}

/**
 * Handle animation speed slider change
 */
function handleSpeedChange() {
    speedValue.textContent = animationSpeedSlider.value;
}

/**
 * Handle custom input apply button click
 */
function handleCustomInputApply() {
    const inputText = customInputField.value.trim();
    if (!inputText) {
        alert('Please enter valid input');
        return;
    }

    try {
        // Parse input: split by commas, remove whitespace, convert to numbers
        const parsedInput = inputText.split(',')
            .map(item => item.trim())
            .filter(item => item !== '')
            .map(item => {
                const num = Number(item);
                if (isNaN(num)) {
                    throw new Error(`"${item}" is not a valid number`);
                }
                return num;
            });

        if (parsedInput.length === 0) {
            throw new Error('No valid numbers found');
        }

        currentInputArray = parsedInput;
        useCustomInput = true;

        // Update the input size slider to match custom input length
        if (parsedInput.length <= parseInt(inputSizeSlider.max)) {
            inputSizeSlider.value = parsedInput.length;
            inputSizeValue.textContent = parsedInput.length;
        }

        // Automatically run algorithm if auto-update is enabled
        if (autoUpdateEnabled && window.runCurrentAlgorithm) {
            window.runCurrentAlgorithm();
        }
    } catch (error) {
        alert(`Invalid input: ${error.message}`);
    }
}

/**
 * Generate a random array of integers
 * @param {number} size - Size of the array to generate
 */
function generateRandomArray(size) {
    currentInputArray = [];
    for (let i = 0; i < size; i++) {
        currentInputArray.push(Math.floor(Math.random() * 100) + 1); // Random ints between 1 and 100
    }
    useCustomInput = false;

    // Update custom input field with the generated array
    customInputField.value = currentInputArray.join(', ');
}

/**
 * Get the current input array
 * @returns {Array<number>} - The current input array
 */
export function getInputArray() {
    return [...currentInputArray]; // Return a copy
}

/**
 * Get the current animation speed
 * @returns {number} - Animation speed (1-10)
 */
export function getAnimationSpeed() {
    return parseInt(animationSpeedSlider.value);
}

/**
 * Reset to random input array
 */
export function resetToRandomInput() {
    useCustomInput = false;
    generateRandomArray(parseInt(inputSizeSlider.value));
    return [...currentInputArray];
}

/**
 * Set a specific input array
 * @param {Array<number>} array - Array to set as current input
 */
export function setInputArray(array) {
    if (Array.isArray(array) && array.length > 0) {
        currentInputArray = [...array];
        useCustomInput = true;
        customInputField.value = currentInputArray.join(', ');

        // Update slider if possible
        if (array.length <= parseInt(inputSizeSlider.max)) {
            inputSizeSlider.value = array.length;
            inputSizeValue.textContent = array.length;
        }
    }
}

/**
 * Get target value for search algorithms
 * @returns {number} - The target value
 */
export function getTargetValue() {
    // By default, use the middle element in the array if possible
    if (currentInputArray.length > 0) {
        return currentInputArray[Math.floor(currentInputArray.length / 2)];
    }
    return 50; // Default target
}

/**
 * Get additional algorithm-specific options
 * @param {string} categoryId - Category ID
 * @param {string} algorithmId - Algorithm ID
 * @returns {Object} - Options for the algorithm
 */
export function getAlgorithmOptions(categoryId, algorithmId) {
    const options = {};

    // Add category-specific options
    if (categoryId === 'searching') {
        options.target = getTargetValue();
    } else if (categoryId === 'graph') {
        options.source = 'A'; // Default source vertex
        options.target = 'E'; // Default target vertex
    }

    return options;
}