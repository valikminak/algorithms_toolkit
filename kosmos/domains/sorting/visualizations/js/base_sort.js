// kosmos/domains/sorting/visualizations/js/base_sort_visualizer.js

class BaseSortVisualizer {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.options = this._mergeDefaults(options);
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.frames = [];
    this.currentFrame = 0;
    this.isPlaying = false;
    this.animationTimer = null;

    // Set up canvas
    this.canvas.width = this.container.clientWidth;
    this.canvas.height = 400;
    this.canvas.classList.add('visualization-canvas');
    this.container.appendChild(this.canvas);

    // Add responsive handling
    window.addEventListener('resize', this._handleResize.bind(this));

    // Create controls
    this._createControls();
  }

  _mergeDefaults(options) {
    return {
      barSpacing: options.barSpacing || 0.2,
      maxBarSpacing: options.maxBarSpacing || 4,
      labelHeight: options.labelHeight || 40,
      animationSpeed: options.animationSpeed || 1,
      colors: {
        bar: options.colors?.bar || '#4a6ee0',
        highlight: options.colors?.highlight || '#e04a6e',
        sorted: options.colors?.sorted || '#28a745',
        background: options.colors?.background || '#f8f9fa',
        text: options.colors?.text || '#333333'
      },
      ...options
    };
  }

  setData(frames) {
    this.frames = frames;
    this.currentFrame = 0;
    this.render();
  }

  render() {
    if (!this.frames.length) {
      this._showPlaceholder("No data to visualize");
      return;
    }

    const frame = this.frames[this.currentFrame];

    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.fillStyle = this.options.colors.background;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw array state
    this._renderArrayState(frame);

    // Update info text
    this._updateInfoText(frame.info || '');
  }

  _renderArrayState(frame) {
    // Abstract method to be implemented by visualization types
  }

  play() {
    if (this.isPlaying) return;

    this.isPlaying = true;
    this._updatePlayButton();

    const frameDelay = 1000 / this.options.animationSpeed;

    this.animationTimer = setInterval(() => {
      this.nextFrame();

      if (this.currentFrame >= this.frames.length - 1) {
        this.pause();
      }
    }, frameDelay);
  }

  pause() {
    if (!this.isPlaying) return;

    this.isPlaying = false;
    clearInterval(this.animationTimer);
    this._updatePlayButton();
  }

  nextFrame() {
    if (this.currentFrame < this.frames.length - 1) {
      this.currentFrame++;
      this.render();
      this._updateProgressBar();
    }
  }

  prevFrame() {
    if (this.currentFrame > 0) {
      this.currentFrame--;
      this.render();
      this._updateProgressBar();
    }
  }

  reset() {
    this.pause();
    this.currentFrame = 0;
    this.render();
    this._updateProgressBar();
  }

  setAnimationSpeed(speed) {
    this.options.animationSpeed = speed;
    if (this.isPlaying) {
      this.pause();
      this.play();
    }
  }

  // Control UI methods...
  _createControls() {
    // Create UI controls for playback
  }

  _updateProgressBar() {
    // Update progress bar UI
  }

  _updatePlayButton() {
    // Update play/pause button UI
  }

  _handleResize() {
    // Handle window resize events
  }

  _updateInfoText(text) {
    // Update info text UI
  }

  _showPlaceholder(message) {
    // Show placeholder when no data is available
  }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = BaseSortVisualizer;
} else {
  window.BaseSortVisualizer = BaseSortVisualizer;
}