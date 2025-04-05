from flask import Flask, render_template, jsonify, send_from_directory
import os
import sys

# Add the parent directory to sys.path to import from algorithm_toolkit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import asset bundler
from utils.asset_bundler import AssetBundler

app = Flask(__name__)

# Configure static folder for bundled assets
app.static_folder = 'static'

# Bundle assets from algorithm domains
bundler = AssetBundler()
bundler.bundle_assets()

# Import routes after app is created to avoid circular imports
from routes.sorting_routes import sorting_bp
from routes.searching_routes import searching_bp
from routes.graph_routes import graph_bp
# Import other route blueprints as needed

# Register blueprints
app.register_blueprint(sorting_bp, url_prefix='/api/sorting')
app.register_blueprint(searching_bp, url_prefix='/api/searching')
app.register_blueprint(graph_bp, url_prefix='/api/graph')
# Register other blueprints

@app.route('/')
def index():
    """Render the main dashboard page"""
    # Get bundled assets for the template
    assets = bundler.get_domain_assets()
    return render_template('index.html', bundled_assets=assets)

@app.route('/api/categories')
def get_categories():
    """Return all algorithm categories"""
    categories = [
        {"id": "sorting", "name": "Sorting Algorithms"},
        {"id": "searching", "name": "Searching Algorithms"},
        {"id": "graph", "name": "Graph Algorithms"},
        {"id": "tree", "name": "Tree Algorithms"},
        {"id": "dynamic-programming", "name": "Dynamic Programming"},
        {"id": "strings", "name": "String Algorithms"},
        {"id": "geometry", "name": "Geometric Algorithms"},
        {"id": "genetic", "name": "Genetic Algorithms"}
    ]
    return jsonify(categories)

@app.route('/bundled/<path:filename>')
def bundled_static(filename):
    """Serve static files from the bundled directory"""
    return send_from_directory(os.path.join(app.static_folder, 'bundled'), filename)

if __name__ == '__main__':
    app.run(debug=True)