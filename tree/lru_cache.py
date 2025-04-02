from typing import Any, Dict, Optional


class DoublyLinkedListNode:
    """Node in a doubly linked list for LRU Cache."""

    def __init__(self, key: Any, value: Any):
        """
        Initialize a doubly linked list node.

        Args:
            key: The key associated with this node
            value: The value stored in this node
        """
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """
    Least Recently Used (LRU) Cache implementation.

    LRU Cache is a cache eviction algorithm where the least recently used items are
    discarded first when the cache reaches its capacity.
    """

    def __init__(self, capacity: int):
        """
        Initialize an LRU Cache with the given capacity.

        Args:
            capacity: Maximum number of key-value pairs to store
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.size = 0
        self.cache: Dict[Any, DoublyLinkedListNode] = {}

        # Dummy head and tail nodes for the doubly linked list
        self.head = DoublyLinkedListNode(None, None)  # Most recently used
        self.tail = DoublyLinkedListNode(None, None)  # Least recently used

        # Connect head and tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: Any) -> Any:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up

        Returns:
            The value associated with the key, or None if not found
        """
        if key not in self.cache:
            return None

        # Move the accessed node to the front (most recently used)
        node = self.cache[key]
        self._move_to_front(node)

        return node.value

    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to insert or update
            value: The value to store
        """
        # If key already exists, update its value and move to front
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
            return

        # If at capacity, remove least recently used item (from tail)
        if self.size >= self.capacity:
            self._remove_lru()

        # Create new node and add to front
        new_node = DoublyLinkedListNode(key, value)
        self.cache[key] = new_node
        self._add_to_front(new_node)
        self.size += 1

    def _add_to_front(self, node: DoublyLinkedListNode) -> None:
        """
        Add a node to the front of the doubly linked list.

        Args:
            node: The node to add
        """
        node.next = self.head.next
        node.prev = self.head

        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DoublyLinkedListNode) -> None:
        """
        Remove a node from the doubly linked list.

        Args:
            node: The node to remove
        """
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_front(self, node: DoublyLinkedListNode) -> None:
        """
        Move a node to the front of the doubly linked list.

        Args:
            node: The node to move
        """
        self._remove_node(node)
        self._add_to_front(node)

    def _remove_lru(self) -> None:
        """Remove the least recently used item from the cache."""
        lru_node = self.tail.prev
        self._remove_node(lru_node)
        del self.cache[lru_node.key]
        self.size -= 1

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def __str__(self) -> str:
        """String representation of the cache."""
        values = []
        current = self.head.next

        while current != self.tail:
            values.append(f"{current.key}: {current.value}")
            current = current.next

        return "LRUCache(capacity={}, size={}, items=[{}])".format(
            self.capacity, self.size, ", ".join(values)
        )


class LFUCache:
    """
    Least Frequently Used (LFU) Cache implementation.

    LFU Cache is a cache eviction algorithm where the least frequently used items are
    discarded first when the cache reaches its capacity. If two items have the same
    frequency, the least recently used one is removed.
    """

    def __init__(self, capacity: int):
        """
        Initialize an LFU Cache with the given capacity.

        Args:
            capacity: Maximum number of key-value pairs to store
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.size = 0
        self.min_freq = 0

        # Dictionary mapping key to (value, frequency) pairs
        self.key_to_val_freq: Dict[Any, tuple[Any, int]] = {}

        # Dictionary mapping frequency to a list of keys with that frequency
        # We maintain the keys in order of insertion (LRU order)
        self.freq_to_keys: Dict[int, list] = {}

    def get(self, key: Any) -> Any:
        """
        Retrieve a value from the cache.

        Args:
            key: The key to look up

        Returns:
            The value associated with the key, or None if not found
        """
        if key not in self.key_to_val_freq:
            return None

        # Increase frequency
        val, freq = self.key_to_val_freq[key]
        self.key_to_val_freq[key] = (val, freq + 1)

        # Remove from current frequency list
        self.freq_to_keys[freq].remove(key)

        # If the list becomes empty and it's the min_freq, increment min_freq
        if not self.freq_to_keys[freq] and self.min_freq == freq:
            self.min_freq += 1

        # Add to new frequency list
        if freq + 1 not in self.freq_to_keys:
            self.freq_to_keys[freq + 1] = []
        self.freq_to_keys[freq + 1].append(key)

        return val

    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update a key-value pair in the cache.

        Args:
            key: The key to insert or update
            value: The value to store
        """
        # If capacity is 0, do nothing
        if self.capacity == 0:
            return

        # If key already exists, update its value and frequency
        if key in self.key_to_val_freq:
            # Update value and increase frequency
            _, freq = self.key_to_val_freq[key]
            self.key_to_val_freq[key] = (value, freq)

            # Call get to handle frequency update
            self.get(key)
            return

        # If at capacity, remove least frequently used item
        if self.size >= self.capacity:
            # Get the key to remove (least frequently used)
            remove_key = self.freq_to_keys[self.min_freq][0]
            self.freq_to_keys[self.min_freq].pop(0)
            del self.key_to_val_freq[remove_key]
            self.size -= 1

        # Add new key with frequency 1
        self.key_to_val_freq[key] = (value, 1)
        if 1 not in self.freq_to_keys:
            self.freq_to_keys[1] = []
        self.freq_to_keys[1].append(key)

        # Update min_freq to 1 since we added a new item
        self.min_freq = 1
        self.size += 1

    def clear(self) -> None:
        """Clear the cache."""
        self.key_to_val_freq.clear()
        self.freq_to_keys.clear()
        self.size = 0
        self.min_freq = 0

    def __str__(self) -> str:
        """String representation of the cache."""
        items = []
        for key, (val, freq) in self.key_to_val_freq.items():
            items.append(f"{key}: ({val}, freq={freq})")

        return "LFUCache(capacity={}, size={}, items=[{}])".format(
            self.capacity, self.size, ", ".join(items)
        )