from typing import List, Optional


class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) implementation for efficient prefix sum queries.

    A Fenwick Tree or Binary Indexed Tree is a data structure that efficiently supports
    prefix sum queries and updates in O(log n) time. It uses a clever bit manipulation
    technique to traverse the tree.
    """

    def __init__(self, size: int):
        """
        Initialize a Fenwick Tree of the given size.

        Args:
            size: Size of the array
        """
        self.size = size
        self.tree = [0] * (size + 1)  # 1-indexed for simplicity of implementation

    @classmethod
    def from_array(cls, arr: List[int]) -> 'FenwickTree':
        """
        Create a Fenwick Tree from an array.

        Args:
            arr: Input array

        Returns:
            Initialized Fenwick Tree
        """
        tree = cls(len(arr))

        # Copy array values
        for i in range(len(arr)):
            tree.update(i, arr[i])

        return tree

    def update(self, index: int, value: int) -> None:
        """
        Add a value to the element at the given index.

        Args:
            index: 0-indexed position
            value: Value to add
        """
        # Convert to 1-indexed
        index += 1

        # Update all relevant nodes in the tree
        while index <= self.size:
            self.tree[index] += value
            index += index & -index  # Add the least significant bit

    def set(self, index: int, value: int) -> None:
        """
        Set the value at the given index.

        Args:
            index: 0-indexed position
            value: New value
        """
        # Get current value
        current = self.get(index)

        # Update with the difference
        self.update(index, value - current)

    def prefix_sum(self, index: int) -> int:
        """
        Compute the sum of elements from 0 to index (inclusive).

        Args:
            index: 0-indexed position

        Returns:
            Sum of elements from 0 to index
        """
        # Convert to 1-indexed
        index += 1
        result = 0

        # Traverse ancestors in the tree
        while index > 0:
            result += self.tree[index]
            index -= index & -index  # Remove the least significant bit

        return result

    def range_sum(self, start: int, end: int) -> int:
        """
        Compute the sum of elements from start to end (inclusive).

        Args:
            start: 0-indexed start position
            end: 0-indexed end position

        Returns:
            Sum of elements from start to end
        """
        if start == 0:
            return self.prefix_sum(end)
        return self.prefix_sum(end) - self.prefix_sum(start - 1)

    def get(self, index: int) -> int:
        """
        Get the value at the given index.

        Args:
            index: 0-indexed position

        Returns:
            Value at the index
        """
        if index == 0:
            return self.prefix_sum(0)
        return self.prefix_sum(index) - self.prefix_sum(index - 1)


class FenwickTree2D:
    """
    2D Fenwick Tree implementation for efficient 2D prefix sum queries.

    This data structure supports efficient sum queries and updates on a 2D grid.
    """

    def __init__(self, rows: int, cols: int):
        """
        Initialize a 2D Fenwick Tree.

        Args:
            rows: Number of rows
            cols: Number of columns
        """
        self.rows = rows
        self.cols = cols
        # 1-indexed for simplicity
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    @classmethod
    def from_matrix(cls, matrix: List[List[int]]) -> 'FenwickTree2D':
        """
        Create a 2D Fenwick Tree from a matrix.

        Args:
            matrix: Input matrix

        Returns:
            Initialized 2D Fenwick Tree
        """
        rows, cols = len(matrix), len(matrix[0]) if matrix else 0
        tree = cls(rows, cols)

        # Copy matrix values
        for i in range(rows):
            for j in range(cols):
                tree.update(i, j, matrix[i][j])

        return tree

    def update(self, row: int, col: int, value: int) -> None:
        """
        Add a value to the element at the given position.

        Args:
            row: 0-indexed row
            col: 0-indexed column
            value: Value to add
        """
        # Convert to 1-indexed
        row += 1
        col += 1

        # Update all relevant nodes in the tree
        i = row
        while i <= self.rows:
            j = col
            while j <= self.cols:
                self.tree[i][j] += value
                j += j & -j  # Add the least significant bit in column
            i += i & -i  # Add the least significant bit in row

    def prefix_sum(self, row: int, col: int) -> int:
        """
        Compute the sum of elements in the rectangle from (0,0) to (row,col).

        Args:
            row: 0-indexed row
            col: 0-indexed column

        Returns:
            Sum of elements in the rectangle
        """
        # Convert to 1-indexed
        row += 1
        col += 1
        result = 0

        # Traverse ancestors in the tree
        i = row
        while i > 0:
            j = col
            while j > 0:
                result += self.tree[i][j]
                j -= j & -j  # Remove the least significant bit in column
            i -= i & -i  # Remove the least significant bit in row

        return result

    def range_sum(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """
        Compute the sum of elements in the rectangle from (row1,col1) to (row2,col2).

        Args:
            row1: 0-indexed top row
            col1: 0-indexed left column
            row2: 0-indexed bottom row
            col2: 0-indexed right column

        Returns:
            Sum of elements in the rectangle
        """
        # Calculate using inclusion-exclusion principle
        result = self.prefix_sum(row2, col2)

        if row1 > 0:
            result -= self.prefix_sum(row1 - 1, col2)
        if col1 > 0:
            result -= self.prefix_sum(row2, col1 - 1)
        if row1 > 0 and col1 > 0:
            result += self.prefix_sum(row1 - 1, col1 - 1)

        return result

    def get(self, row: int, col: int) -> int:
        """
        Get the value at the given position.

        Args:
            row: 0-indexed row
            col: 0-indexed column

        Returns:
            Value at the position
        """
        return self.range_sum(row, col, row, col)

    def set(self, row: int, col: int, value: int) -> None:
        """
        Set the value at the given position.

        Args:
            row: 0-indexed row
            col: 0-indexed column
            value: New value
        """
        current = self.get(row, col)
        self.update(row, col, value - current)


class FenwickTreeRangeUpdate:
    """
    Fenwick Tree with support for range updates and point queries.

    This is an extension of the standard Fenwick Tree that efficiently supports
    updates on a range of elements and queries for individual elements.
    """

    def __init__(self, size: int):
        """
        Initialize a Fenwick Tree for range updates.

        Args:
            size: Size of the array
        """
        self.size = size
        # Use two BITs to support range updates
        self.tree1 = [0] * (size + 1)  # Stores b[i] * i
        self.tree2 = [0] * (size + 1)  # Stores b[i]

    def _update(self, tree: List[int], index: int, value: int) -> None:
        """
        Helper function to update a specific Fenwick Tree.

        Args:
            tree: The Fenwick Tree to update
            index: 1-indexed position
            value: Value to add
        """
        while index <= self.size:
            tree[index] += value
            index += index & -index

    def _query(self, tree: List[int], index: int) -> int:
        """
        Helper function to query a specific Fenwick Tree.

        Args:
            tree: The Fenwick Tree to query
            index: 1-indexed position

        Returns:
            Prefix sum up to index
        """
        result = 0
        while index > 0:
            result += tree[index]
            index -= index & -index
        return result

    def range_update(self, start: int, end: int, value: int) -> None:
        """
        Add a value to all elements in the range [start, end].

        Args:
            start: 0-indexed start position
            end: 0-indexed end position
            value: Value to add
        """
        # Convert to 1-indexed
        start += 1
        end += 1

        # Update first tree
        self._update(self.tree1, start, value * (start - 1))
        self._update(self.tree1, end + 1, -value * end)

        # Update second tree
        self._update(self.tree2, start, value)
        self._update(self.tree2, end + 1, -value)

    def point_query(self, index: int) -> int:
        """
        Get the value at the given index.

        Args:
            index: 0-indexed position

        Returns:
            Value at the index
        """
        # Convert to 1-indexed
        index += 1
        return self._query(self.tree1, index) - self._query(self.tree2, index) * (index - 1)


class FenwickTreeRangeQuery:
    """
    Fenwick Tree with support for range queries and point updates.

    This is the standard Fenwick Tree implementation optimized for
    efficient range sum queries.
    """

    def __init__(self, size: int):
        """
        Initialize a Fenwick Tree for range queries.

        Args:
            size: Size of the array
        """
        self.size = size
        self.tree = [0] * (size + 1)

    @classmethod
    def from_array(cls, arr: List[int]) -> 'FenwickTreeRangeQuery':
        """
        Create a Fenwick Tree from an array.

        Args:
            arr: Input array

        Returns:
            Initialized Fenwick Tree
        """
        tree = cls(len(arr))

        # Copy array values
        for i in range(len(arr)):
            tree.point_update(i, arr[i])

        return tree

    def point_update(self, index: int, value: int) -> None:
        """
        Add a value to the element at the given index.

        Args:
            index: 0-indexed position
            value: Value to add
        """
        # Convert to 1-indexed
        index += 1

        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        """
        Compute the sum of elements from 0 to index (inclusive).

        Args:
            index: 0-indexed position

        Returns:
            Sum of elements from 0 to index
        """
        # Convert to 1-indexed
        index += 1
        result = 0

        while index > 0:
            result += self.tree[index]
            index -= index & -index

        return result

    def range_query(self, start: int, end: int) -> int:
        """
        Compute the sum of elements from start to end (inclusive).

        Args:
            start: 0-indexed start position
            end: 0-indexed end position

        Returns:
            Sum of elements from start to end
        """
        if start == 0:
            return self.prefix_sum(end)
        return self.prefix_sum(end) - self.prefix_sum(start - 1)