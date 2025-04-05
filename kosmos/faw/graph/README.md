# Graph Algorithm Visualization Setup Guide

## Installation

1. Ensure all files are in the correct directory structure:

```
├── graph/
│   ├── __init__.py
│   ├── base.py
│   ├── traversal.py
│   └── shortest_path.py
├── web/
│   ├── app.py
│   ├── routes/
│   │   ├── graph_routes.py
│   │   ├── searching_routes.py
│   │   └── sorting_routes.py
│   ├── static/
│   │   ├── css/
│   │   │   ├── normalize.css
│   │   │   ├── style.css
│   │   │   └── graph-ui.css
│   │   └── js/
│   │       ├── algorithms.js
│   │       ├── graph-ui.js
│   │       ├── graph-visualization.js
│   │       ├── main.js
│   │       ├── ui-controls.js
│   │       └── visualization.js
│   └── templates/
│       └── index.html
└── utils/
    ├── performance.py
    └── visualization.py
```

2. Make sure you have all required Python packages installed:

```bash
pip install flask matplotlib networkx
```

## Running the Application

1. Start the Flask app by running:

```bash
cd /path/to/your/project
python web/app.py
```

2. Open your browser and navigate to:

```
http://localhost:5000
```

## Using the Graph Visualization

1. **Select the "Graph Algorithms" category** from the sidebar
   - The graph editor will appear automatically below the controls panel

2. **Create a graph**
   - Manually: 
     - Double-click in the empty area to add a vertex
     - Click the "Add Edge" button, then click on two vertices to create an edge
     - Double-click on vertices or edges to edit their labels/weights
   - Or select an example graph from the dropdown menu

3. **Configure algorithm parameters**
   - Select a start vertex from the "Start Vertex" dropdown
   - Select a target vertex from the "Target Vertex" dropdown (for pathfinding algorithms)
   - Choose directed/undirected and weighted/unweighted options as needed

4. **Select an algorithm** from the algorithm dropdown
   - BFS (Breadth-First Search)
   - DFS (Depth-First Search)
   - Dijkstra's Algorithm
   - A* Search
   - Prim's MST
   - Kruskal's MST

5. **Click "Run"** to start the visualization

6. **Use the animation controls**
   - Play/Pause: Start or pause the animation
   - Step Forward/Backward: Move through the algorithm steps manually
   - Restart: Go back to the beginning
   - Speed: Adjust the animation speed with the slider in the controls panel

## Understanding the Visualization

- **Node Colors:**
  - Blue: Default/unvisited node
  - Green: Visited node
  - Yellow: Node in the queue/frontier
  - Red: Currently active node
  - Orange: Node in the final path
  - Purple: Start vertex
  - Orange: Target vertex

- **Edge Colors:**
  - Gray: Default edge
  - Yellow: Edge being considered
  - Orange: Edge in the final path/MST

- **Algorithm Information:**
  - The info text below the visualization explains what's happening at each step
  - For Dijkstra's algorithm, distance labels are shown next to the nodes

## Troubleshooting

If you encounter issues:

1. Check the browser console for JavaScript errors
2. Verify that all files are in the correct locations
3. Make sure the Flask server is running without errors
4. Try refreshing the page if the graph editor doesn't appear
5. If the visualization doesn't work, try selecting a different algorithm or reloading the page