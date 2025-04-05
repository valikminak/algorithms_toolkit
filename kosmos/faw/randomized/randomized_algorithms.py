from typing import List, Any, TypeVar, Generic, Tuple, Dict
import random

T = TypeVar('T')


def randomized_quickselect(arr: List[T], k: int) -> T:
    """
    Randomized QuickSelect algorithm to find the kth smallest element in an array.

    This algorithm has O(n) expected time complexity.

    Args:
        arr: Input array
        k: The rank of the element to find (1-indexed)

    Returns:
        The kth smallest element in the array
    """
    if not arr or k < 1 or k > len(arr):
        raise ValueError("Invalid input or k value")

    # Make a copy to avoid modifying the original array
    arr_copy = arr.copy()

    # Convert to 0-indexed
    k = k - 1

    return _quickselect(arr_copy, 0, len(arr_copy) - 1, k)


def _quickselect(arr: List[T], left: int, right: int, k: int) -> T:
    """
    Helper function for randomized quickselect.

    Args:
        arr: Input array
        left: Left boundary index
        right: Right boundary index
        k: Index of the element to find (0-indexed)

    Returns:
        The kth smallest element
    """
    if left == right:
        return arr[left]

    # Choose a random pivot
    pivot_idx = random.randint(left, right)

    # Move pivot to the right
    arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]

    # Partition around the pivot
    pivot_idx = _partition(arr, left, right)

    if k == pivot_idx:
        return arr[k]
    elif k < pivot_idx:
        return _quickselect(arr, left, pivot_idx - 1, k)
    else:
        return _quickselect(arr, pivot_idx + 1, right, k)


def _partition(arr: List[T], left: int, right: int) -> int:
    """
    Partition the array around the pivot (last element).

    Args:
        arr: Input array
        left: Left boundary index
        right: Right boundary index

    Returns:
        Final position of the pivot
    """
    pivot = arr[right]
    i = left - 1

    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


def reservoir_sampling(stream, k: int) -> List[Any]:
    """
    Reservoir sampling algorithm to select k random items from a stream of unknown size.

    This algorithm ensures each item in the stream has an equal probability of being selected.

    Args:
        stream: Iterator yielding items from the stream
        k: Number of items to sample

    Returns:
        List of k randomly selected items from the stream
    """
    reservoir = []
    n = 0

    for item in stream:
        n += 1
        if n <= k:
            reservoir.append(item)
        else:
            # Randomly replace items with decreasing probability
            j = random.randint(0, n - 1)
            if j < k:
                reservoir[j] = item

    return reservoir


def monte_carlo_pi_estimation(num_points: int) -> float:
    """
    Estimate the value of π using Monte Carlo simulation.

    This algorithm randomly samples points in a square and counts how many fall
    within a quarter circle to estimate π.

    Args:
        num_points: Number of random points to generate

    Returns:
        Estimation of π
    """
    inside_circle = 0

    for _ in range(num_points):
        # Generate random point in the unit square
        x = random.random()
        y = random.random()

        # Check if point is inside the quarter circle
        if x * x + y * y <= 1:
            inside_circle += 1

    # Ratio of points inside quarter circle is π/4
    return 4 * inside_circle / num_points


def miller_rabin_primality_test(n: int, k: int = 40) -> bool:
    """
    Miller-Rabin primality test - a probabilistic algorithm to determine if a number is prime.

    Args:
        n: The number to test for primality
        k: Number of iterations (higher k increases accuracy)

    Returns:
        True if n is probably prime, False if n is definitely composite
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


class SkipListNode:
    """Node in a Skip List."""

    def __init__(self, level: int, value: Any):
        """
        Initialize a Skip List node.

        Args:
            level: The level of the node (number of forward pointers)
            value: The value stored in the node
        """
        self.value = value
        self.forward = [None] * (level + 1)


class SkipList(Generic[T]):
    """
    Skip List implementation - a probabilistic alternative to balanced trees.

    Skip List is a data structure that allows O(log n) search, insertion, and deletion
    in an ordered sequence of elements, using a probabilistic approach.
    """

    def __init__(self, max_level: int = 16, p: float = 0.5):
        """
        Initialize an empty Skip List.

        Args:
            max_level: Maximum level of the Skip List
            p: Probability factor for the level of a new node
        """
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.head = SkipListNode(max_level, None)

    def _random_level(self) -> int:
        """
        Randomly determine the level of a new node.

        Returns:
            Random level between 0 and max_level
        """
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def search(self, value: T) -> bool:
        """
        Search for a value in the Skip List.

        Args:
            value: The value to search for

        Returns:
            True if the value is found, False otherwise
        """
        current = self.head

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]

        # Current is smaller than the value, so check the next node
        current = current.forward[0]

        # If current is not None and has the target value, it's found
        return current is not None and current.value == value

    def insert(self, value: T) -> None:
        """
        Insert a value into the Skip List.

        Args:
            value: The value to insert
        """
        # Update array to track nodes to update at each level
        update = [None] * (self.max_level + 1)
        current = self.head

        # Find position to insert
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]
            update[i] = current

        # Get a random level for the new node
        level = self._random_level()

        # Update the Skip List level if necessary
        if level > self.level:
            for i in range(self.level + 1, level + 1):
                update[i] = self.head
            self.level = level

        # Create the new node
        new_node = SkipListNode(level, value)

        # Insert the node at each level
        for i in range(level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def delete(self, value: T) -> bool:
        """
        Delete a value from the Skip List.

        Args:
            value: The value to delete

        Returns:
            True if the value was deleted, False if not found
        """
        # Update array to track nodes to update at each level
        update = [None] * (self.max_level + 1)
        current = self.head

        # Find position to delete
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]
            update[i] = current

        # Move to the next node
        current = current.forward[0]

        # If current node has the target value, delete it
        if current and current.value == value:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            # Update the level if necessary
            while self.level > 0 and self.head.forward[self.level] is None:
                self.level -= 1

            return True

        return False

    def __str__(self) -> str:
        """String representation of the Skip List."""
        result = []

        # For each level
        for i in range(self.level + 1):
            current = self.head.forward[i]
            level_str = f"Level {i}: "

            while current:
                level_str += f"{current.value} -> "
                current = current.forward[i]

            level_str += "None"
            result.append(level_str)

        return "\n".join(result)


def shuffle_array(arr: List[T]) -> List[T]:
    """
    Randomly shuffle an array using the Fisher-Yates algorithm.

    This algorithm guarantees that each permutation is equally likely.

    Args:
        arr: Input array

    Returns:
        Shuffled array
    """
    # Make a copy to avoid modifying the original array
    result = arr.copy()
    n = len(result)

    for i in range(n - 1, 0, -1):
        # Choose a random index from 0 to i
        j = random.randint(0, i)

        # Swap elements at i and j
        result[i], result[j] = result[j], result[i]

    return result


def randomized_min_cut(graph: Dict[int, List[int]]) -> Tuple[int, List[List[int]]]:
    """
    Karger's randomized min-cut algorithm for finding the minimum cut of a graph.

    Args:
        graph: Adjacency list representation of an undirected graph

    Returns:
        Tuple of (cut_size, partitions) where:
        - cut_size: Size of the minimum cut
        - partitions: List of two vertex sets forming the cut
    """
    # Create a copy of the graph
    g = {u: v.copy() for u, v in graph.items()}

    # Create vertex groups (initially each vertex in its own group)
    groups = {u: [u] for u in g}

    while len(g) > 2:
        # Choose a random edge
        u = random.choice(list(g.keys()))
        v = random.choice(g[u])

        # Merge v into u
        groups[u].extend(groups[v])
        del groups[v]

        # Update edges
        for w in g[v]:
            if w != u:  # Skip self-loops
                g[w].remove(v)
                g[w].append(u)
                g[u].append(w)

        # Remove v from graph
        del g[v]

        # Remove self-loops
        g[u] = [w for w in g[u] if w != u]

    # Get the two remaining vertex groups
    partitions = list(groups.values())

    # Count edges between the two groups (min cut size)
    u, v = list(g.keys())
    cut_size = len(g[u])  # Number of edges from u to v

    return cut_size, partitions