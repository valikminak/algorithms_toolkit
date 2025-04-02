from typing import List, Callable, Optional, Any, Union
import math


class SegmentTree:
    """
    Segment Tree implementation for efficient range queries.

    A Segment Tree is a binary tree used for storing intervals or segments. It allows
    querying which of the stored segments contain a given point or overlap with a given range.
    """

    def __init__(self, arr: List[Any], operation: Callable[[Any, Any], Any], identity_element: Any):
        """
        Initialize a segment tree for the given array.

        Args:
            arr: Input array for which segment tree is built
            operation: Function that defines how to combine elements (e.g., min, max, sum)
            identity_element: Identity element for the operation (e.g., inf for min, -inf for max, 0 for sum)
        """
        self.n = len(arr)
        self.operation = operation
        self.identity_element = identity_element

        # Size of segment tree
        x = math.ceil(math.log2(self.n))
        max_size = 2 * (2 ** x) - 1

        # Initialize with identity element
        self.tree = [identity_element] * max_size

        if self.n > 0:
            self._build_tree(arr, 0, self.n - 1, 0)

    def _build_tree(self, arr: List[Any], start: int, end: int, index: int) -> Any:
        """
        Recursive function to build the segment tree.

        Args:
            arr: Input array
            start: Start index of current segment
            end: End index of current segment
            index: Index in the segment tree

        Returns:
            Value stored at the current node
        """
        # Leaf node
        if start == end:
            self.tree[index] = arr[start]
            return self.tree[index]

        # Recursive build left and right
        mid = (start + end) // 2
        left = self._build_tree(arr, start, mid, 2 * index + 1)
        right = self._build_tree(arr, mid + 1, end, 2 * index + 2)

        # Combine results
        self.tree[index] = self.operation(left, right)

        return self.tree[index]

    def update(self, arr_index: int, new_value: Any) -> None:
        """
        Update the value at the given array index.

        Args:
            arr_index: Index in the original array
            new_value: New value to set
        """
        if arr_index < 0 or arr_index >= self.n:
            raise IndexError("Index out of bounds")

        self._update_tree(0, self.n - 1, arr_index, new_value, 0)

    def _update_tree(self, start: int, end: int, arr_index: int, new_value: Any, index: int) -> None:
        """
        Recursive function to update the segment tree.

        Args:
            start: Start index of current segment
            end: End index of current segment
            arr_index: Index to update in the original array
            new_value: New value to set
            index: Index in the segment tree
        """
        # Out of range
        if arr_index < start or arr_index > end:
            return

        # Leaf node
        if start == end:
            self.tree[index] = new_value
            return

        # Recur for left and right children
        mid = (start + end) // 2
        self._update_tree(start, mid, arr_index, new_value, 2 * index + 1)
        self._update_tree(mid + 1, end, arr_index, new_value, 2 * index + 2)

        # Update current node
        self.tree[index] = self.operation(self.tree[2 * index + 1], self.tree[2 * index + 2])

    def query(self, query_start: int, query_end: int) -> Any:
        """
        Query the segment tree for a range.

        Args:
            query_start: Start of the query range
            query_end: End of the query range

        Returns:
            Result of the operation applied to the range
        """
        if query_start < 0 or query_end >= self.n or query_start > query_end:
            raise ValueError("Invalid query range")

        return self._query_tree(0, self.n - 1, query_start, query_end, 0)

    def _query_tree(self, start: int, end: int, query_start: int, query_end: int, index: int) -> Any:
        """
        Recursive function to query the segment tree.

        Args:
            start: Start index of current segment
            end: End index of current segment
            query_start: Start of query range
            query_end: End of query range
            index: Index in the segment tree

        Returns:
            Result of the operation applied to the range
        """
        # Segment is completely inside the query range
        if query_start <= start and query_end >= end:
            return self.tree[index]

        # Segment is completely outside the query range
        if end < query_start or start > query_end:
            return self.identity_element

        # Partial overlap - query both children
        mid = (start + end) // 2
        left_result = self._query_tree(start, mid, query_start, query_end, 2 * index + 1)
        right_result = self._query_tree(mid + 1, end, query_start, query_end, 2 * index + 2)

        return self.operation(left_result, right_result)


class LazySegmentTree:
    """
    Lazy Segment Tree implementation with lazy propagation for efficient range updates.

    This implementation allows efficient range updates by propagating updates lazily.
    """

    def __init__(self, arr: List[Any], operation: Callable[[Any, Any], Any],
                 update_operation: Callable[[Any, Any, int], Any],
                 identity_element: Any, lazy_identity: Any):
        """
        Initialize a lazy segment tree for the given array.

        Args:
            arr: Input array for which segment tree is built
            operation: Function that defines how to combine elements (e.g., min, max, sum)
            update_operation: Function to update segment with lazy value and segment length
            identity_element: Identity element for the operation
            lazy_identity: Identity element for lazy updates
        """
        self.n = len(arr)
        self.operation = operation
        self.update_operation = update_operation
        self.identity_element = identity_element
        self.lazy_identity = lazy_identity

        # Size of segment tree
        x = math.ceil(math.log2(self.n))
        max_size = 2 * (2 ** x) - 1

        # Initialize tree and lazy array
        self.tree = [identity_element] * max_size
        self.lazy = [lazy_identity] * max_size

        if self.n > 0:
            self._build_tree(arr, 0, self.n - 1, 0)

    def _build_tree(self, arr: List[Any], start: int, end: int, index: int) -> Any:
        """
        Recursive function to build the segment tree.

        Args:
            arr: Input array
            start: Start index of current segment
            end: End index of current segment
            index: Index in the segment tree

        Returns:
            Value stored at the current node
        """
        # Leaf node
        if start == end:
            self.tree[index] = arr[start]
            return self.tree[index]

        # Recursive build left and right
        mid = (start + end) // 2
        left = self._build_tree(arr, start, mid, 2 * index + 1)
        right = self._build_tree(arr, mid + 1, end, 2 * index + 2)

        # Combine results
        self.tree[index] = self.operation(left, right)

        return self.tree[index]

    def _lazy_propagate(self, start: int, end: int, index: int) -> None:
        """
        Propagate lazy updates to children.

        Args:
            start: Start index of current segment
            end: End index of current segment
            index: Index in the segment tree
        """
        # If this is a leaf node or has identity lazy value, no propagation needed
        if start == end or self.lazy[index] == self.lazy_identity:
            return

        # Apply lazy update to current node
        segment_length = end - start + 1
        self.tree[index] = self.update_operation(self.tree[index], self.lazy[index], segment_length)

        # Propagate to children if not a leaf
        if start != end:
            # Update lazy values of children
            left_child = 2 * index + 1
            right_child = 2 * index + 2

            if self.lazy[left_child] == self.lazy_identity:
                self.lazy[left_child] = self.lazy[index]
            else:
                # Combine lazy updates - depends on the specific update operation
                self.lazy[left_child] += self.lazy[index]  # Example for sum updates

            if self.lazy[right_child] == self.lazy_identity:
                self.lazy[right_child] = self.lazy[index]
            else:
                # Combine lazy updates
                self.lazy[right_child] += self.lazy[index]  # Example for sum updates

        # Reset lazy value for this node
        self.lazy[index] = self.lazy_identity

    def update_range(self, update_start: int, update_end: int, value: Any) -> None:
        """
        Update a range of values in the tree.

        Args:
            update_start: Start of the update range
            update_end: End of the update range
            value: Value to update with
        """
        if update_start < 0 or update_end >= self.n or update_start > update_end:
            raise ValueError("Invalid update range")

        self._update_range_tree(0, self.n - 1, update_start, update_end, value, 0)

    def _update_range_tree(self, start: int, end: int, update_start: int, update_end: int,
                           value: Any, index: int) -> None:
        """
        Recursive function to update a range in the segment tree.

        Args:
            start: Start index of current segment
            end: End index of current segment
            update_start: Start of update range
            update_end: End of update range
            value: Value to update with
            index: Index in the segment tree
        """
        # Propagate lazy updates first
        self._lazy_propagate(start, end, index)

        # Segment is completely outside the update range
        if end < update_start or start > update_end:
            return

        # Segment is completely inside the update range
        if update_start <= start and update_end >= end:
            # Update this node
            segment_length = end - start + 1
            self.tree[index] = self.update_operation(self.tree[index], value, segment_length)

            # If not leaf, mark children for lazy propagation
            if start != end:
                left_child = 2 * index + 1
                right_child = 2 * index + 2

                if self.lazy[left_child] == self.lazy_identity:
                    self.lazy[left_child] = value
                else:
                    # Combine lazy updates
                    self.lazy[left_child] += value  # Example for sum updates

                if self.lazy[right_child] == self.lazy_identity:
                    self.lazy[right_child] = value
                else:
                    # Combine lazy updates
                    self.lazy[right_child] += value  # Example for sum updates

            return

        # Partial overlap - update both children
        mid = (start + end) // 2
        self._update_range_tree(start, mid, update_start, update_end, value, 2 * index + 1)
        self._update_range_tree(mid + 1, end, update_start, update_end, value, 2 * index + 2)

        # Combine results
        self.tree[index] = self.operation(self.tree[2 * index + 1], self.tree[2 * index + 2])

    def query(self, query_start: int, query_end: int) -> Any:
        """
        Query the segment tree for a range.

        Args:
            query_start: Start of the query range
            query_end: End of the query range

        Returns:
            Result of the operation applied to the range
        """
        if query_start < 0 or query_end >= self.n or query_start > query_end:
            raise ValueError("Invalid query range")

        return self._query_tree(0, self.n - 1, query_start, query_end, 0)

    def _query_tree(self, start: int, end: int, query_start: int, query_end: int, index: int) -> Any:
        """
        Recursive function to query the segment tree.

        Args:
            start: Start index of current segment
            end: End index of current segment
            query_start: Start of query range
            query_end: End of query range
            index: Index in the segment tree

        Returns:
            Result of the operation applied to the range
        """
        # Propagate lazy updates first
        self._lazy_propagate(start, end, index)

        # Segment is completely outside the query range
        if end < query_start or start > query_end:
            return self.identity_element

        # Segment is completely inside the query range
        if query_start <= start and query_end >= end:
            return self.tree[index]

        # Partial overlap - query both children
        mid = (start + end) // 2
        left_result = self._query_tree(start, mid, query_start, query_end, 2 * index + 1)
        right_result = self._query_tree(mid + 1, end, query_start, query_end, 2 * index + 2)

        return self.operation(left_result, right_result)


# Example usage:
def create_sum_segment_tree(arr: List[int]) -> SegmentTree:
    """Create a segment tree for sum queries."""
    return SegmentTree(arr, lambda a, b: a + b, 0)


def create_min_segment_tree(arr: List[int]) -> SegmentTree:
    """Create a segment tree for minimum queries."""
    return SegmentTree(arr, min, float('inf'))


def create_max_segment_tree(arr: List[int]) -> SegmentTree:
    """Create a segment tree for maximum queries."""
    return SegmentTree(arr, max, float('-inf'))


def create_gcd_segment_tree(arr: List[int]) -> SegmentTree:
    """Create a segment tree for GCD queries."""

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    return SegmentTree(arr, gcd, 0)


def create_lazy_sum_segment_tree(arr: List[int]) -> LazySegmentTree:
    """Create a lazy segment tree for sum queries with range updates."""

    def update_sum(current, value, length):
        return current + value * length

    return LazySegmentTree(arr, lambda a, b: a + b, update_sum, 0, 0)