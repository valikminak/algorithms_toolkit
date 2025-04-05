class SortingVisualizer {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container with ID "${containerId}" not found`);
      return;
    }

    this.options = this._mergeDefaults(options);
    this.frames = [];
    this.currentFrame = 0;
    this.animationTimer = null;
    this.isPlaying = false;

    // Create canvas
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.container.clientWidth;
    this.canvas.height = this.options.height;
    this.container.appendChild(this.canvas);

    // Get context
    this.ctx = this.canvas.getContext('2d');

    // Create info element
    this.infoElement = document.createElement('div');
    this.infoElement.className = 'text-center font-medium mt-4';
    this.container.appendChild(this.infoElement);

    // Create progress bar
    this.progressBarContainer = document.createElement('div');
    this.progressBarContainer.className = 'progress-bar mt-4';
    this.progressBarFill = document.createElement('div');
    this.progressBarFill.className = 'progress-bar-fill';
    this.progressBarFill.style.width = '0%';
    this.progressBarContainer.appendChild(this.progressBarFill);
    this.container.appendChild(this.progressBarContainer);

    // Create controls
    this.controls = document.createElement('div');
    this.controls.className = 'flex justify-center gap-2 mt-4';
    this.controls.innerHTML = `
      <button class="button-sm restart" title="Restart">⏮️</button>
      <button class="button-sm prev" title="Previous Step">⏪</button>
      <button class="button-sm play-pause" title="Play/Pause">▶️</button>
      <button class="button-sm next" title="Next Step">⏩</button>
    `;
    this.container.appendChild(this.controls);

    // Add event listeners
    this._setupEventListeners();

    // Handle window resize
    window.addEventListener('resize', this._handleResize.bind(this));
  }

  _mergeDefaults(options) {
    return {
      height: options.height || 300,
      speed: options.speed || 1,
      barSpacing: options.barSpacing || 0.2,
      maxBarSpacing: options.maxBarSpacing || 4,
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

  _setupEventListeners() {
    // Get control buttons
    const playPauseBtn = this.controls.querySelector('.play-pause');
    const nextBtn = this.controls.querySelector('.next');
    const prevBtn = this.controls.querySelector('.prev');
    const restartBtn = this.controls.querySelector('.restart');

    // Add event listeners
    playPauseBtn.addEventListener('click', () => {
      if (this.isPlaying) {
        this.pause();
      } else {
        this.play();
      }
    });

    nextBtn.addEventListener('click', () => this.nextFrame());
    prevBtn.addEventListener('click', () => this.prevFrame());
    restartBtn.addEventListener('click', () => this.restart());

    // Progress bar click handler
    this.progressBarContainer.addEventListener('click', (e) => {
      const rect = this.progressBarContainer.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      const frameIndex = Math.floor(ratio * this.frames.length);
      this.goToFrame(frameIndex);
    });
  }

  _handleResize() {
    // Resize canvas
    this.canvas.width = this.container.clientWidth;

    // Redraw current frame
    this.render();
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

    // Set background
    this.ctx.fillStyle = this.options.colors.background;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Get array state
    const array = frame.state || [];

    // Skip if empty
    if (array.length === 0) {
      this._showPlaceholder("Empty array");
      return;
    }

    // Find max value
    const maxValue = Math.max(...array.map(v => isNaN(v) ? 0 : v));

    // Calculate bar width
    const barWidth = this.canvas.width / array.length;
    const barSpacing = Math.min(barWidth * this.options.barSpacing, this.options.maxBarSpacing);

    // Draw array bars
    for (let i = 0; i < array.length; i++) {
      const value = array[i];
      const barHeight = (value / maxValue) * (this.canvas.height - 40);

      // Determine bar color
      let color = this.options.colors.bar;

      // Highlight elements being compared/swapped
      if (frame.highlight && frame.highlight.includes(i)) {
        color = this.options.colors.highlight;
      }

      // Draw bar
      this.ctx.fillStyle = color;
      this.ctx.fillRect(
        i * barWidth + barSpacing / 2,
        this.canvas.height - barHeight,
        barWidth - barSpacing,
        barHeight
      );

      // Draw value if there's enough space
      if (barWidth > 20) {
        this.ctx.fillStyle = this.options.colors.text;
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';
        this.ctx.fillText(
          value.toString(),
          i * barWidth + barWidth / 2,
          this.canvas.height - barHeight - 20
        );
      }
    }

    // Update info text
    this.infoElement.textContent = frame.info || '';

    // Update progress bar
    const progress = (this.currentFrame / (this.frames.length - 1)) * 100;
    this.progressBarFill.style.width = `${progress}%`;

    // Add stats
    this.ctx.fillStyle = this.options.colors.text;
    this.ctx.font = '12px Arial';
    this.ctx.textAlign = 'left';
    this.ctx.fillText(`Comparisons: ${frame.comparisons || 0}`, 10, 20);
    this.ctx.fillText(`Swaps: ${frame.swaps || 0}`, 10, 40);
    this.ctx.fillText(`Frame: ${this.currentFrame + 1}/${this.frames.length}`, 10, 60);
  }

  _showPlaceholder(message) {
    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw background
    this.ctx.fillStyle = this.options.colors.background;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw message
    this.ctx.fillStyle = '#666';
    this.ctx.font = '14px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(message, this.canvas.width / 2, this.canvas.height / 2);

    // Clear info text
    this.infoElement.textContent = '';
  }

  play() {
    if (this.isPlaying) return;

    this.isPlaying = true;
    this.controls.querySelector('.play-pause').textContent = '⏸️';

    // Calculate delay based on speed
    const delay = 1000 / this.options.speed;

    // Start animation
    this.animationTimer = setInterval(() => {
      this.nextFrame();

      // Stop at end
      if (this.currentFrame >= this.frames.length - 1) {
        this.pause();
      }
    }, delay);
  }

  pause() {
    if (!this.isPlaying) return;

    this.isPlaying = false;
    clearInterval(this.animationTimer);
    this.controls.querySelector('.play-pause').textContent = '▶️';
  }

  nextFrame() {
    if (this.currentFrame < this.frames.length - 1) {
      this.currentFrame++;
      this.render();
    }
  }

  prevFrame() {
    if (this.currentFrame > 0) {
      this.currentFrame--;
      this.render();
    }
  }

  restart() {
    this.pause();
    this.currentFrame = 0;
    this.render();
  }

  goToFrame(frameIndex) {
    this.pause();
    this.currentFrame = Math.max(0, Math.min(frameIndex, this.frames.length - 1));
    this.render();
  }

  setSpeed(speed) {
    this.options.speed = speed;

    // Restart animation if playing
    if (this.isPlaying) {
      this.pause();
      this.play();
    }
  }
}

window.SortingVisualizer = SortingVisualizer;