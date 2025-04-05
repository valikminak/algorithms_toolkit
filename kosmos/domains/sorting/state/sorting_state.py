# kosmos/domains/sorting/state/sorting_state.py
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SortingState:
    """Represents the state of a sorting algorithm at a point in time."""

    array: List[int]
    current_indices: List[int]
    comparison_count: int
    swap_count: int
    info_text: Optional[str] = None

    def clone(self) -> 'SortingState':
        """Create a deep copy of this state."""
        return SortingState(
            array=self.array.copy(),
            current_indices=self.current_indices.copy(),
            comparison_count=self.comparison_count,
            swap_count=self.swap_count,
            info_text=self.info_text
        )

    def __post_init__(self):
        if self.current_indices is None:
            self.current_indices = []