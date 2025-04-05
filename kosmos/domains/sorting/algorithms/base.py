from abc import abstractmethod
from typing import List, Dict

from kosmos.core.algorithm import Algorithm
from kosmos.domains.sorting.state.state_manager import SortingStateManager


class SortingAlgorithm(Algorithm[List[int], List[int]]):
    """Base class for all sorting algorithms."""

    def __init__(self):
        self.state_manager = SortingStateManager()

    def execute(self, input_data: List[int]) -> List[int]:
        """Execute sorting algorithm with state recording."""
        # Make a copy to avoid modifying the input
        arr = input_data.copy()

        # Initialize state manager
        self.state_manager.initialize(arr)

        # Execute algorithm-specific sort
        self._sort(arr)

        return arr

    @abstractmethod
    def _sort(self, arr: List[int]) -> None:
        """Algorithm-specific sorting implementation."""

    def get_visualization_frames(self, input_data: List[int]) -> List[Dict]:
        """Generate visualization frames from algorithm execution."""
        # Execute the algorithm to generate state history
        self.execute(input_data)

        # Convert state history to frames
        return self.state_manager.get_frames()