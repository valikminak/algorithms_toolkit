from kosmos.core.state_manager import StateManager
from kosmos.domains.sorting.state.sorting_state import SortingState
from typing import List, Dict


class SortingStateManager(StateManager[SortingState]):
    """Manages state for sorting algorithms."""

    def create_initial_state(self, input_data: List[int]) -> SortingState:
        """Create initial sorting state."""
        return SortingState(
            array=input_data.copy(),
            current_indices=[],
            comparison_count=0,
            swap_count=0,
            info_text="Initial state"
        )

    def _state_to_frame(self, state: SortingState) -> Dict:
        """Convert sorting state to a visualization frame."""
        return {
            "state": state.array.copy(),
            "highlight": state.current_indices.copy() if state.current_indices else [],
            "info": state.info_text or "",
            "comparisons": state.comparison_count,
            "swaps": state.swap_count
        }

    def compare(self, i: int, j: int) -> bool:
        """Record a comparison between elements at indices i and j."""
        if self.current_state is None:
            raise ValueError("State not initialized")

        result = self.current_state.array[i] > self.current_state.array[j]

        def updater(state: SortingState):
            state.current_indices = [i, j]
            state.comparison_count += 1
            state.info_text = f"Comparing {state.array[i]} and {state.array[j]}"

        self.update_state(updater)
        return result

    def swap(self, i: int, j: int) -> None:
        """Record and perform a swap between elements at indices i and j."""
        if self.current_state is None:
            raise ValueError("State not initialized")

        def updater(state: SortingState):
            state.current_indices = [i, j]
            state.swap_count += 1
            state.info_text = f"Swapping {state.array[i]} and {state.array[j]}"
            # Perform the swap
            state.array[i], state.array[j] = state.array[j], state.array[i]

        self.update_state(updater)

    def mark_sorted(self, indices: List[int]) -> None:
        """Mark elements as sorted."""
        if self.current_state is None:
            raise ValueError("State not initialized")

        def updater(state: SortingState):
            state.current_indices = indices.copy()
            state.info_text = f"Elements at positions {indices} are now sorted"

        self.update_state(updater)

    def set_info(self, text: str) -> None:
        """Set information text."""
        if self.current_state is None:
            raise ValueError("State not initialized")

        def updater(state: SortingState):
            state.info_text = text

        self.update_state(updater)