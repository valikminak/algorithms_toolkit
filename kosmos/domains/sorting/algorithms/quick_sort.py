from typing import List, Dict
from kosmos.domains.sorting.algorithms.base import SortingAlgorithm


class QuickSort(SortingAlgorithm):
    """
    Quick Sort implementation.

    Time Complexity:
        - Best: O(n log n)
        - Average: O(n log n)
        - Worst: O(n²) - can happen with poor pivot selection

    Space Complexity: O(log n) - for recursion stack

    Stable: No
    """

    def _sort(self, arr: List[int]) -> None:
        """
        Sort an array using quick sort algorithm.

        This modifies the array in-place.
        """
        self._quick_sort(arr, 0, len(arr) - 1)

    def _quick_sort(self, arr: List[int], low: int, high: int) -> None:
        """Recursive quick sort implementation."""
        if low < high:
            # Partition and get pivot index
            pivot_idx = self._partition(arr, low, high)

            # Sort left part
            self._quick_sort(arr, low, pivot_idx - 1)

            # Sort right part
            self._quick_sort(arr, pivot_idx + 1, high)

    def _partition(self, arr: List[int], low: int, high: int) -> int:
        """Partition the array and return the pivot index."""
        # Choose last element as pivot
        pivot = arr[high]
        self.state_manager.set_info(f"Partitioning with pivot {pivot}")

        i = low - 1  # Index of smaller element

        for j in range(low, high):
            # Compare with pivot
            if not self.state_manager.compare(j, high):  # arr[j] <= pivot
                i += 1
                # Swap elements
                if i != j:
                    self.state_manager.swap(i, j)
                    arr[i], arr[j] = arr[j], arr[i]

        # Place pivot in correct position
        i += 1
        self.state_manager.swap(i, high)
        arr[i], arr[high] = arr[high], arr[i]

        # Mark pivot as in correct position
        self.state_manager.mark_sorted([i])
        self.state_manager.set_info(f"Pivot {pivot} is in the correct position")

        return i

    @property
    def metadata(self) -> Dict:
        return {
            "id": "quick_sort",
            "name": "Quick Sort",
            "category": "sorting",
            "time_complexity": {
                "best": "O(n log n)",
                "average": "O(n log n)",
                "worst": "O(n²)"
            },
            "space_complexity": "O(log n)",
            "stable": False,
            "description": "A divide-and-conquer algorithm that works by selecting a 'pivot' element and partitioning the array around the pivot."
        }