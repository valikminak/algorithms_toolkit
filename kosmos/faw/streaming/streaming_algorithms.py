from typing import List, Dict, Any, Tuple
import math
import random
import hashlib
from collections import defaultdict


class CountMinSketch:
    """
    Count-Min Sketch for approximate frequency counting in a data stream.

    This is a probabilistic data structure that serves as a frequency table of elements
    in a stream of data, using sub-linear space.
    """

    def __init__(self, epsilon: float = 0.01, delta: float = 0.01):
        """
        Initialize a Count-Min Sketch.

        Args:
            epsilon: Error bound (controls width of the table)
            delta: Probability of error exceeding the bound (controls depth of the table)
        """
        self.width = math.ceil(math.e / epsilon)
        self.depth = math.ceil(math.log(1 / delta))
        self.table = [[0] * self.width for _ in range(self.depth)]

        # Initialize hash functions (using simple universal hashing)
        self.hash_params = []
        for _ in range(self.depth):
            # Generate random parameters for hash function
            a = random.randint(1, 1000000007)
            b = random.randint(0, 1000000007)
            self.hash_params.append((a, b))

    def _hash(self, x: Any, i: int) -> int:
        """
        Compute the ith hash function value for x.

        Args:
            x: The item to hash
            i: The index of the hash function

        Returns:
            Hash value in the range [0, width-1]
        """
        a, b = self.hash_params[i]
        hash_val = (a * hash(x) + b) % 1000000007
        return hash_val % self.width

    def update(self, x: Any, count: int = 1) -> None:
        """
        Update the frequency of an item in the sketch.

        Args:
            x: The item to update
            count: Amount to increment the count by
        """
        for i in range(self.depth):
            j = self._hash(x, i)
            self.table[i][j] += count

    def estimate(self, x: Any) -> int:
        """
        Estimate the frequency of an item.

        Args:
            x: The item to estimate

        Returns:
            Estimated frequency
        """
        return min(self.table[i][self._hash(x, i)] for i in range(self.depth))

    def merge(self, other: 'CountMinSketch') -> None:
        """
        Merge another Count-Min Sketch into this one.

        Args:
            other: Another Count-Min Sketch with the same dimensions
        """
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Can only merge sketches with the same dimensions")

        for i in range(self.depth):
            for j in range(self.width):
                self.table[i][j] += other.table[i][j]


class HyperLogLog:
    """
    HyperLogLog algorithm for cardinality estimation (counting unique elements).

    This probabilistic algorithm estimates the number of distinct elements in a stream
    with a very small memory footprint.
    """

    def __init__(self, precision: int = 14):
        """
        Initialize a HyperLogLog counter.

        Args:
            precision: Number of bits used for register addressing (controls accuracy)
        """
        if precision < 4 or precision > 16:
            raise ValueError("Precision must be between 4 and 16")

        self.precision = precision
        self.num_registers = 1 << precision
        self.registers = [0] * self.num_registers
        self.alpha = self._get_alpha(self.num_registers)

    def _get_alpha(self, m: int) -> float:
        """
        Calculate the alpha constant based on the number of registers.

        Args:
            m: Number of registers

        Returns:
            Alpha value for bias correction
        """
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)

    def _get_register_and_rank(self, x: Any) -> Tuple[int, int]:
        """
        Get the register index and the rank (position of the first 1-bit) for an item.

        Args:
            x: The item to process

        Returns:
            Tuple of (register_index, rank)
        """
        # Get a hash of the item
        hash_val = int(hashlib.md5(str(x).encode()).hexdigest(), 16)

        # Extract register index from the first precision bits
        register_idx = hash_val & (self.num_registers - 1)

        # Shift right by precision bits and count the position of the first 1-bit
        w = hash_val >> self.precision
        rank = 1
        while (w & 1) == 0 and rank <= 32:
            w >>= 1
            rank += 1

        return register_idx, rank

    def add(self, x: Any) -> None:
        """
        Add an item to the HyperLogLog counter.

        Args:
            x: The item to add
        """
        reg_idx, rank = self._get_register_and_rank(x)
        self.registers[reg_idx] = max(self.registers[reg_idx], rank)

    def count(self) -> float:
        """
        Estimate the number of distinct items added.

        Returns:
            Estimated cardinality
        """
        # Calculate the harmonic mean of the registers
        sum_inv = sum(math.pow(2, -r) for r in self.registers)

        # Estimate the cardinality
        estimate = self.alpha * self.num_registers * self.num_registers / sum_inv

        # Apply corrections for small and large estimates
        if estimate <= 2.5 * self.num_registers:
            # Count number of zeros in registers
            num_zeros = self.registers.count(0)
            if num_zeros > 0:
                # Linear counting for small ranges
                return self.num_registers * math.log(self.num_registers / num_zeros)

        return estimate

    def merge(self, other: 'HyperLogLog') -> None:
        """
        Merge another HyperLogLog counter into this one.

        Args:
            other: Another HyperLogLog counter with the same precision
        """
        if self.precision != other.precision:
            raise ValueError("Can only merge HyperLogLog counters with the same precision")

        for i in range(self.num_registers):
            self.registers[i] = max(self.registers[i], other.registers[i])


class BloomFilter:
    """
    Bloom Filter for space-efficient set membership testing.

    A Bloom filter is a probabilistic data structure that tests whether an element
    is a member of a set. False positives are possible, but false negatives are not.
    """

    def __init__(self, capacity: int, error_rate: float = 0.01):
        """
        Initialize a Bloom Filter.

        Args:
            capacity: Expected number of elements to be added
            error_rate: Desired false positive rate
        """
        # Calculate optimal number of bits and hash functions
        self.size = self._calculate_size(capacity, error_rate)
        self.hash_count = self._calculate_hash_count(self.size, capacity)

        # Initialize bit array
        self.bit_array = [False] * self.size

    def _calculate_size(self, capacity: int, error_rate: float) -> int:
        """
        Calculate the optimal size of the bit array.

        Args:
            capacity: Expected number of elements
            error_rate: Desired false positive rate

        Returns:
            Optimal size in bits
        """
        size = -capacity * math.log(error_rate) / (math.log(2) ** 2)
        return math.ceil(size)

    def _calculate_hash_count(self, size: int, capacity: int) -> int:
        """
        Calculate the optimal number of hash functions.

        Args:
            size: Size of the bit array in bits
            capacity: Expected number of elements

        Returns:
            Optimal number of hash functions
        """
        k = size / capacity * math.log(2)
        return math.ceil(k)

    def _get_hash_values(self, item: Any) -> List[int]:
        """
        Get the hash values for an item.

        Args:
            item: The item to hash

        Returns:
            List of hash values
        """
        # Use a single hash function with different seeds for efficiency
        value = str(item).encode()
        h1 = int(hashlib.md5(value).hexdigest(), 16)
        h2 = int(hashlib.sha1(value).hexdigest(), 16)

        # Generate hash_count different hash values
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]

    def add(self, item: Any) -> None:
        """
        Add an item to the Bloom Filter.

        Args:
            item: The item to add
        """
        for idx in self._get_hash_values(item):
            self.bit_array[idx] = True

    def contains(self, item: Any) -> bool:
        """
        Check if an item might be in the Bloom Filter.

        Args:
            item: The item to check

        Returns:
            True if the item might be in the set, False if it is definitely not
        """
        for idx in self._get_hash_values(item):
            if not self.bit_array[idx]:
                return False
        return True


class SlidingWindowCounter:
    """
    Sliding Window Counter for tracking statistics over a sliding time window.

    This data structure maintains counts of items within a sliding window,
    allowing efficient queries for recent data.
    """

    def __init__(self, window_size: int, bucket_count: int = 10):
        """
        Initialize a Sliding Window Counter.

        Args:
            window_size: Size of the sliding window in time units
            bucket_count: Number of buckets to divide the window into
        """
        if bucket_count <= 0 or window_size <= 0:
            raise ValueError("Window size and bucket count must be positive")

        self.window_size = window_size
        self.bucket_count = bucket_count
        self.bucket_size = window_size / bucket_count

        # Initialize buckets
        self.buckets = [defaultdict(int) for _ in range(bucket_count)]
        self.current_bucket = 0
        self.last_timestamp = 0

    def add(self, item: Any, timestamp: int, count: int = 1) -> None:
        """
        Add an item at the given timestamp.

        Args:
            item: The item to add
            timestamp: Current timestamp
            count: Count to add
        """
        # Update buckets based on the time elapsed
        self._update_buckets(timestamp)

        # Add the item to the current bucket
        self.buckets[self.current_bucket][item] += count

        # Update last timestamp
        self.last_timestamp = timestamp

    def _update_buckets(self, timestamp: int) -> None:
        """
        Update buckets based on the time elapsed since last update.

        Args:
            timestamp: Current timestamp
        """
        # Calculate how many buckets to advance
        if timestamp < self.last_timestamp:
            raise ValueError("Timestamps must be non-decreasing")

        time_diff = timestamp - self.last_timestamp
        buckets_to_advance = int(time_diff / self.bucket_size)

        if buckets_to_advance >= self.bucket_count:
            # If we need to advance more than the total number of buckets, clear all
            self.buckets = [defaultdict(int) for _ in range(self.bucket_count)]
            self.current_bucket = 0
        elif buckets_to_advance > 0:
            # Advance buckets and clear the old ones
            for i in range(buckets_to_advance):
                next_bucket = (self.current_bucket + 1) % self.bucket_count
                self.buckets[next_bucket] = defaultdict(int)
                self.current_bucket = next_bucket

    def count(self, item: Any) -> int:
        """
        Get the count of an item within the sliding window.

        Args:
            item: The item to count

        Returns:
            Total count of the item in the window
        """
        return sum(bucket[item] for bucket in self.buckets)

    def get_counts(self) -> Dict[Any, int]:
        """
        Get counts of all items in the window.

        Returns:
            Dictionary mapping items to their counts
        """
        result = defaultdict(int)
        for bucket in self.buckets:
            for item, count in bucket.items():
                result[item] += count
        return dict(result)