from typing import List, Dict
from kosmos.domains.sorting.algorithms.base import SortingAlgorithm


class BubbleSort(SortingAlgorithm):
    """
    Bubble Sort implementation.

    Time Complexity:
        - Best: O(n) when array is already sorted
        - Average: O(n²)
        - Worst: O(n²)

    Space Complexity: O(1)

    Stable: Yes
    """

    def _sort(self, arr: List[int]) -> None:
        """
        Sort an array using bubble sort algorithm.

        This modifies the array in-place.
        """
        n = len(arr)

        for i in range(n):
            swapped = False

            for j in range(0, n - i - 1):
                # Compare elements
                if self.state_manager.compare(j, j + 1):
                    # Swap elements
                    self.state_manager.swap(j, j + 1)
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True

            # Mark element as in correct position
            self.state_manager.mark_sorted([n - i - 1])

            # Early termination if no swaps
            if not swapped:
                self.state_manager.set_info("Array is sorted (early termination)")
                break

        # Final state
        self.state_manager.set_info("Array is fully sorted")

    @property
    def metadata(self) -> Dict:
        return {
            "id": "bubble_sort",
            "name": "Bubble Sort",
            "category": "sorting",
            "time_complexity": {
                "best": "O(n)",
                "average": "O(n²)",
                "worst": "O(n²)"
            },
            "space_complexity": "O(1)",
            "stable": True,
            "description": "A simple comparison-based sorting algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order."
        }