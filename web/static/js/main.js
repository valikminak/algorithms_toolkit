document.addEventListener('DOMContentLoaded', () => {
    return Alpine.data('algorithmApp', () => ({
        // State
        domains: [],
        algorithms: [],
        selectedDomain: null,
        selectedAlgorithm: null,
        inputArray: [5, 3, 8, 1, 2, 9],
        customInput: '5, 3, 8, 1, 2, 9',
        arraySize: 10,
        animationSpeed: 3,
        isRunning: false,
        error: null,
        activeTab: 'visualization',

        // Visualizer instance
        visualizer: null,

        // Initialize
        async init() {
            // Load domains
            await this.loadDomains();

            // Setup visualizer
            this.visualizer = new SortingVisualizer('visualization-container', {
                speed: this.animationSpeed
            });

            // Apply default inputs
            this.generateRandomArray();
        },

        // Load domains from API
        async loadDomains() {
            try {
                const response = await fetch('/api/domains');
                const data = await response.json();
                this.domains = data.domains;

                // Auto-select first domain
                if (this.domains.length > 0) {
                    this.selectDomain(this.domains[0]);
                }
            } catch (error) {
                console.error('Error loading domains:', error);
                this.error = 'Failed to load algorithm domains';
            }
        },

        // Load algorithms for selected domain
        async loadAlgorithms(domainId) {
            try {
                const response = await fetch(`/api/domains/${domainId}/algorithms`);
                const data = await response.json();
                this.algorithms = data.algorithms;

                // Auto-select first algorithm
                if (this.algorithms.length > 0) {
                    this.selectAlgorithm(this.algorithms[0].id);
                }
            } catch (error) {
                console.error(`Error loading algorithms for ${domainId}:`, error);
                this.error = `Failed to load algorithms for ${domainId}`;
                this.algorithms = [];
            }
        },

        // Select a domain
        async selectDomain(domainId) {
            this.selectedDomain = domainId;
            this.selectedAlgorithm = null;
            await this.loadAlgorithms(domainId);
        },

        // Select an algorithm
        selectAlgorithm(algorithmId) {
            this.selectedAlgorithm = algorithmId;
        },

        // Run the selected algorithm
        async runAlgorithm() {
            if (!this.selectedDomain || !this.selectedAlgorithm) {
                this.error = 'Please select an algorithm';
                return;
            }

            this.isRunning = true;
            this.error = null;

            try {
                const response = await fetch(
                    `/api/domains/${this.selectedDomain}/algorithms/${this.selectedAlgorithm}/run`,
                    {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            input: this.inputArray,
                            options: {}
                        })
                    }
                );

                const result = await response.json();

                // Update visualizer with frames
                if (result.visualization && this.visualizer) {
                    this.visualizer.setData(result.visualization);
                    this.visualizer.setSpeed(this.animationSpeed);
                    this.visualizer.play();
                }

                // Switch to visualization tab
                this.activeTab = 'visualization';
            } catch (error) {
                console.error('Error running algorithm:', error);
                this.error = 'Failed to run algorithm';
            } finally {
                this.isRunning = false;
            }
        },

        // Generate random array
        generateRandomArray() {
            const array = [];
            for (let i = 0; i < this.arraySize; i++) {
                array.push(Math.floor(Math.random() * 100) + 1);
            }
            this.inputArray = array;
            this.customInput = array.join(', ');
        },

        // Apply custom input
        applyCustomInput() {
            try {
                const array = this.customInput.split(',')
                    .map(item => item.trim())
                    .filter(item => item !== '')
                    .map(item => {
                        const num = parseInt(item);
                        if (isNaN(num)) {
                            throw new Error(`Invalid number: ${item}`);
                        }
                        return num;
                    });

                if (array.length === 0) {
                    throw new Error('No valid numbers found');
                }

                this.inputArray = array;
                this.arraySize = array.length;
            } catch (error) {
                this.error = `Invalid input: ${error.message}`;
            }
        },

        // Update animation speed
        updateSpeed() {
            if (this.visualizer) {
                this.visualizer.setSpeed(this.animationSpeed);
            }
        },

        // Switch tabs
        switchTab(tab) {
            this.activeTab = tab;
        },

        // Clear error
        clearError() {
            this.error = null;
        }
    }))
});