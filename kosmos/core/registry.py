# kosmos/core/registry.py
import importlib
import inspect
import os
from typing import List

from kosmos.core.algorithm import Algorithm


class AlgorithmRegistry:
    """Registry for algorithm discovery and instantiation."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AlgorithmRegistry, cls).__new__(cls)
            cls._instance._algorithms = {}
            cls._instance._algorithm_classes = {}
            cls._instance._domains = []
            cls._instance._initialized = False
        return cls._instance

    def initialize(self) -> None:
        """Discover and register all algorithms."""
        if self._initialized:
            return

        # Auto-discover domains
        domains_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "domains")

        if not os.path.exists(domains_path):
            print(f"Warning: Domains directory not found at {domains_path}")
            self._initialized = True
            return

        for domain in os.listdir(domains_path):
            domain_path = os.path.join(domains_path, domain)

            # Skip files and special directories
            if not os.path.isdir(domain_path) or domain.startswith('__'):
                continue

            # Add domain to list
            self._domains.append(domain)
            self._algorithms[domain] = {}
            self._algorithm_classes[domain] = {}

            # Look for algorithm implementations
            algorithms_path = os.path.join(domain_path, "algorithms")
            if not os.path.exists(algorithms_path):
                continue

            # Try to import all algorithm modules
            for filename in os.listdir(algorithms_path):
                if not filename.endswith('.py') or filename.startswith('__'):
                    continue

                module_name = f"kosmos.domains.{domain}.algorithms.{filename[:-3]}"

                try:
                    module = importlib.import_module(module_name)

                    # Find all algorithm classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                                issubclass(obj, Algorithm) and
                                obj != Algorithm and
                                not name.startswith('_')):
                            # Register the algorithm
                            algo_id = name.lower()
                            self._algorithm_classes[domain][algo_id] = obj
                except Exception as e:
                    print(f"Error importing {module_name}: {e}")

        self._initialized = True

    def get_domains(self) -> List[str]:
        """Get all available algorithm domains."""
        self.initialize()
        return self._domains

    def get_algorithm_ids(self, domain: str) -> List[str]:
        """Get all algorithm IDs for a domain."""
        self.initialize()
        if domain not in self._algorithm_classes:
            return []
        return list(self._algorithm_classes[domain].keys())

    def get_algorithm(self, domain: str, algorithm_id: str) -> Algorithm:
        """Get an algorithm instance."""
        self.initialize()

        if domain not in self._algorithm_classes:
            raise KeyError(f"Domain {domain} not found")

        if algorithm_id not in self._algorithm_classes[domain]:
            raise KeyError(f"Algorithm {algorithm_id} not found in domain {domain}")

        # Create instance if not already instantiated
        if algorithm_id not in self._algorithms[domain]:
            self._algorithms[domain][algorithm_id] = self._algorithm_classes[domain][algorithm_id]()

        return self._algorithms[domain][algorithm_id]