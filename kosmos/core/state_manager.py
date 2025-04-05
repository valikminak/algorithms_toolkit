from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Callable, Optional, Any
from copy import deepcopy

T = TypeVar('T')  # State type


class StateManager(Generic[T], ABC):
    """Manages algorithm state transitions and visualization frames."""

    def __init__(self):
        self.current_state: Optional[T] = None
        self.state_history: List[T] = []

    @abstractmethod
    def create_initial_state(self, input_data: Any) -> T:
        """Create initial state from input data."""

    def initialize(self, input_data: Any) -> None:
        """Initialize with input data."""
        self.current_state = self.create_initial_state(input_data)
        self.state_history = [deepcopy(self.current_state)]

    def update_state(self, updater: Callable[[T], None]) -> None:
        """Update state using a function and record the new state."""
        if self.current_state is None:
            raise ValueError("State not initialized")

        # Create a new state by deep copying current
        new_state = deepcopy(self.current_state)

        # Apply the update function
        updater(new_state)

        # Save as current state
        self.current_state = new_state

        # Add to history
        self.state_history.append(deepcopy(new_state))

    def get_frames(self) -> List[Dict]:
        """Convert state history to visualization frames."""
        return [self._state_to_frame(state) for state in self.state_history]

    @abstractmethod
    def _state_to_frame(self, state: T) -> Dict:
        """Convert a state object to a visualization frame."""
