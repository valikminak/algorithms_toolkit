import shutil
import json
from pathlib import Path
import sys


class AssetBundler:
    """
    Utility for collecting and managing static assets from algorithm domains.
    This allows each algorithm domain to have its own visualization assets while
    making them available to the Flask application.
    """

    def __init__(self, output_path='web/static/bundled'):
        self.kosmos_path = Path("../kosmos")
        self.output_path = Path(output_path)
        self.manifest = {}

    def discover_domains(self):
        """Find all algorithm domains with visualization assets"""
        domains = []

        # Find all directories in kosmos
        for item in self.kosmos_path.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                # Check if this domain has visualization assets
                vis_path = item / 'visualization'
                if vis_path.exists() and vis_path.is_dir():
                    domains.append(item.name)

        return domains

    def ensure_output_dirs(self):
        """Ensure output directories exist"""
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / 'css').mkdir(exist_ok=True)
        (self.output_path / 'js').mkdir(exist_ok=True)

    def copy_assets(self, domain):
        """Copy assets from a specific domain to the output directory"""
        domain_path = self.kosmos_path / domain
        vis_path = domain_path / 'visualization'

        if not vis_path.exists():
            return

        # Keep track of assets for this domain
        domain_assets = {
            'css': [],
            'js': []
        }

        # Copy CSS files
        for css_file in vis_path.glob('*.css'):
            dest_path = self.output_path / 'css' / f"{domain}_{css_file.name}"
            shutil.copy2(css_file, dest_path)
            domain_assets['css'].append(f"bundled/css/{domain}_{css_file.name}")

        # Copy JS files
        for js_file in vis_path.glob('*.js'):
            dest_path = self.output_path / 'js' / f"{domain}_{js_file.name}"
            shutil.copy2(js_file, dest_path)
            domain_assets['js'].append(f"bundled/js/{domain}_{js_file.name}")

        # Store in manifest
        self.manifest[domain] = domain_assets

    def bundle_assets(self):
        """Bundle all visualization assets from algorithm domains"""
        self.ensure_output_dirs()

        # Clear existing files
        for item in (self.output_path / 'css').glob('*'):
            if item.is_file():
                item.unlink()

        for item in (self.output_path / 'js').glob('*'):
            if item.is_file():
                item.unlink()

        # Discover and copy assets from each domain
        domains = self.discover_domains()
        for domain in domains:
            self.copy_assets(domain)

        # Write manifest
        with open(self.output_path / 'manifest.json', 'w') as f:
            json.dump(self.manifest, f, indent=2)

        print(f"Bundled assets from {len(domains)} domains")

    def get_domain_assets(self, domain=None):
        """Get assets for a specific domain or all domains"""
        try:
            with open(self.output_path / 'manifest.json', 'r') as f:
                self.manifest = json.load(f)

            if domain:
                return self.manifest.get(domain, {'css': [], 'js': []})
            else:
                # Combine all assets
                all_assets = {'css': [], 'js': []}
                for domain_assets in self.manifest.values():
                    all_assets['css'].extend(domain_assets['css'])
                    all_assets['js'].extend(domain_assets['js'])
                return all_assets
        except (FileNotFoundError, json.JSONDecodeError):
            return {'css': [], 'js': []}

    def inject_assets_into_template(self, template_path, output_path=None):
        """
        Modify a template to include domain assets. If output_path is not provided,
        the template will be modified in place.
        """
        if output_path is None:
            output_path = template_path

        with open(template_path, 'r') as f:
            content = f.read()

        # Read manifest
        try:
            with open(self.output_path / 'manifest.json', 'r') as f:
                self.manifest = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.manifest = {}

        # Prepare CSS and JS includes
        css_includes = []
        js_includes = []

        for domain, assets in self.manifest.items():
            for css in assets['css']:
                css_includes.append(f'<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'{css}\') }}">')

            for js in assets['js']:
                js_includes.append(f'<script src="{{ url_for(\'static\', filename=\'{js}\') }}"></script>')

        # Insert includes at appropriate places
        if '<!-- BUNDLED_CSS -->' in content:
            content = content.replace('<!-- BUNDLED_CSS -->', '\n    '.join(css_includes))

        if '<!-- BUNDLED_JS -->' in content:
            content = content.replace('<!-- BUNDLED_JS -->', '\n    '.join(js_includes))

        with open(output_path, 'w') as f:
            f.write(content)

        print(f"Injected {len(css_includes)} CSS and {len(js_includes)} JS files into {output_path}")


def main():
    """Command line interface for the asset bundler"""
    bundler = AssetBundler()

    if len(sys.argv) < 2:
        print("Usage: python asset_bundler.py [bundle|inject template_path]")
        return

    command = sys.argv[1]

    if command == 'bundle':
        bundler.bundle_assets()
    elif command == 'inject' and len(sys.argv) >= 3:
        template_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) >= 4 else None
        bundler.inject_assets_into_template(template_path, output_path)
    else:
        print("Unknown command")


if __name__ == '__main__':
    main()