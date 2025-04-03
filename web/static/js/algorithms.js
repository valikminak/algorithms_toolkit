// algorithms.js - Handles API calls to retrieve and execute algorithms

/**
 * Fetch all algorithm categories from the server
 * @returns {Promise<Array>} - Array of category objects
 */
export async function fetchCategories() {
    try {
        const response = await fetch('/api/categories');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching categories:', error);
        throw error;
    }
}

/**
 * Fetch algorithms for a specific category
 * @param {string} categoryId - ID of the category to fetch algorithms for
 * @returns {Promise<Array>} - Array of algorithm objects
 */
export async function fetchAlgorithms(categoryId) {
    try {
        const response = await fetch(`/api/${categoryId}/algorithms`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching algorithms for ${categoryId}:`, error);
        // Return empty array in case of error to avoid breaking the UI
        return [];
    }
}

/**
 * Run an algorithm with the provided input
 * @param {string} categoryId - Category of the algorithm
 * @param {string} algorithmId - ID of the algorithm to run
 * @param {Array} input - Input data for the algorithm
 * @param {Object} options - Additional options
 * @returns {Promise<Object>} - Result of the algorithm execution
 */
export async function runAlgorithm(categoryId, algorithmId, input, options = {}) {
    try {
        const requestData = {
            algorithm: algorithmId,
            input: input,
            ...options
        };

        const response = await fetch(`/api/${categoryId}/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`Error running algorithm ${algorithmId}:`, error);
        throw error;
    }
}

/**
 * Compare multiple algorithms
 * @param {string} categoryId - Category of algorithms to compare
 * @param {Array<string>} algorithmIds - IDs of algorithms to compare
 * @param {Array} input - Input data for the algorithms
 * @param {Object} options - Additional options
 * @returns {Promise<Object>} - Comparison results
 */
export async function compareAlgorithms(categoryId, algorithmIds, input, options = {}) {
    try {
        const requestData = {
            algorithms: algorithmIds,
            input: input,
            ...options
        };

        const response = await fetch(`/api/${categoryId}/compare`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`Error comparing algorithms:`, error);
        throw error;
    }
}

/**
 * Fetch the implementation code for an algorithm
 * @param {string} categoryId - Category of the algorithm
 * @param {string} algorithmId - ID of the algorithm
 * @returns {Promise<Object>} - Object containing the code
 */
export async function fetchAlgorithmCode(categoryId, algorithmId) {
    try {
        const response = await fetch(`/api/${categoryId}/code?algorithm=${algorithmId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching code for ${algorithmId}:`, error);
        return { code: 'Failed to load code' };
    }
}