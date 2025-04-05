from abc import ABC, abstractmethod
from typing import Dict, List, TypeVar, Generic

from kosmos.api.schemas import AlgorithmMetadata

T = TypeVar('T')  # Input type
U = TypeVar('U')  # Output type


class Algorithm(Generic[T, U], ABC):
    """Base class for all algorithms with visualization support."""

    @abstractmethod
    def execute(self, input_data: T) -> U:
        """Execute the algorithm on input data and return the result."""

    @abstractmethod
    def get_visualization_frames(self, input_data: T) -> List[Dict]:
        """Generate frames for visualization of algorithm execution."""

    @property
    @abstractmethod
    def metadata(self) -> AlgorithmMetadata:
        """Return algorithm metadata (complexity, category, etc)."""
