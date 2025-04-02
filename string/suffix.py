from typing import List, Dict, Set, Tuple, Optional, Any, Union
import collections


class SuffixTrieNode:
    """Node in a Suffix Trie."""

    def __init__(self):
        """Initialize an empty node."""
        self.children = {}  # Maps characters to child nodes
        self.is_end_of_word = False
        self.positions = []  # Positions where this suffix appears in the text


class SuffixTrie:
    """
    Suffix Trie implementation.

    A suffix trie contains all suffixes of a string. This is a simple
    but memory-inefficient implementation compared to suffix trees or arrays.
    """

    def __init__(self, text: str = None):
        """
        Initialize a Suffix Trie, optionally with a text.

        Args:
            text: The text to build the suffix trie from
        """
        self.root = SuffixTrieNode()
        if text:
            self.build(text)

    def build(self, text: str) -> None:
        """
        Build the suffix trie from a text.

        Args:
            text: The text to build the suffix trie from
        """
        # Add all suffixes to the trie
        for i in range(len(text)):
            self._insert_suffix(text[i:], i)

    def _insert_suffix(self, suffix: str, position: int) -> None:
        """
        Insert a suffix into the trie.

        Args:
            suffix: The suffix to insert
            position: The position in the original text where the suffix starts
        """
        node = self.root

        for char in suffix:
            if char not in node.children:
                node.children[char] = SuffixTrieNode()
            node = node.children[char]
            node.positions.append(position)

        node.is_end_of_word = True

    def search(self, pattern: str) -> List[int]:
        """
        Search for all occurrences of a pattern in the text.

        Args:
            pattern: The pattern to search for

        Returns:
            List of positions where the pattern occurs
        """
        node = self.root

        # Traverse the trie to find the pattern
        for char in pattern:
            if char not in node.children:
                return []
            node = node.children[char]

        # Return all positions where the pattern occurs
        return node.positions


class SuffixTreeNode:
    """Node in a Suffix Tree."""

    def __init__(self, start: int, end: Optional[int]):
        """
        Initialize a node with edge information.

        Args:
            start: Start index of the substring on the edge leading to this node
            end: End index of the substring (exclusive), or None for a leaf node
        """
        self.start = start
        self.end = end
        self.children = {}  # Maps characters to child nodes
        self.suffix_link = None
        self.positions = []  # Positions where this suffix appears in the text


class SuffixTree:
    """
    Suffix Tree implementation using Ukkonen's algorithm.

    A suffix tree is a compressed trie representing all suffixes of a string.
    It allows for efficient pattern matching operations.
    """

    def __init__(self, text: str = None):
        """
        Initialize a Suffix Tree, optionally with a text.

        Args:
            text: The text to build the suffix tree from
        """
        self.root = SuffixTreeNode(0, 0)
        self.text = ""
        self.active_node = self.root
        self.active_edge = -1
        self.active_length = 0
        self.remaining = 0
        self.leaf_end = -1
        self.position = -1

        if text:
            self.build(text)

    def build(self, text: str) -> None:
        """
        Build the suffix tree from a text using Ukkonen's algorithm.

        Args:
            text: The text to build the suffix tree from
        """
        self.text = text + "$"  # Add a unique terminator
        self.position = -1
        self.leaf_end = -1

        for i in range(len(self.text)):
            self._extend(i)

    def _extend(self, pos: int) -> None:
        """
        Extend the suffix tree with the character at the given position.

        Args:
            pos: Position in the text to extend with
        """
        self.leaf_end = pos
        self.remaining += 1
        self.position = pos

        # Extension for each suffix
        last_internal_node = None

        while self.remaining > 0:
            if self.active_length == 0:
                self.active_edge = pos

            if self.text[self.active_edge] not in self.active_node.children:
                # Create a new leaf node
                self.active_node.children[self.text[self.active_edge]] = SuffixTreeNode(pos, None)
                self.active_node.children[self.text[self.active_edge]].positions.append(pos - self.remaining + 1)

                # Set suffix link for the last internal node
                if last_internal_node is not None:
                    last_internal_node.suffix_link = self.active_node
                    last_internal_node = None
            else:
                # Check if we can move down the tree
                next_node = self.active_node.children[self.text[self.active_edge]]
                edge_length = self._edge_length(next_node)

                # If active length is within the edge, just update active length
                if self.active_length >= edge_length:
                    self.active_edge += edge_length
                    self.active_length -= edge_length
                    self.active_node = next_node
                    continue

                # Check if current character is already on the edge
                if self.text[next_node.start + self.active_length] == self.text[pos]:
                    self.active_length += 1

                    # Set suffix link for the last internal node
                    if last_internal_node is not None:
                        last_internal_node.suffix_link = self.active_node

                    break

                # Split the edge and create a new internal node
                split_node = SuffixTreeNode(next_node.start, next_node.start + self.active_length)
                self.active_node.children[self.text[self.active_edge]] = split_node

                # Create a leaf node for the new suffix
                split_node.children[self.text[pos]] = SuffixTreeNode(pos, None)
                split_node.children[self.text[pos]].positions.append(pos - self.remaining + 1)

                # Adjust the start index for the old node
                next_node.start += self.active_length
                split_node.children[self.text[next_node.start]] = next_node

                # Set suffix link for the last internal node
                if last_internal_node is not None:
                    last_internal_node.suffix_link = split_node

                last_internal_node = split_node

            self.remaining -= 1

            # Rule 1: If active_node is root, decrease active_length and adjust active_edge
            if self.active_node == self.root and self.active_length > 0:
                self.active_length -= 1
                self.active_edge = pos - self.remaining + 1

            # Rule 3: Follow suffix link if not at root
            elif self.active_node != self.root:
                self.active_node = self.active_node.suffix_link if self.active_node.suffix_link else self.root

    def _edge_length(self, node: SuffixTreeNode) -> int:
        """
        Get the length of the edge leading to the node.

        Args:
            node: The node to get edge length for

        Returns:
            Length of the edge
        """
        if node.end is None:
            return self.leaf_end - node.start + 1
        return node.end - node.start

    def search(self, pattern: str) -> List[int]:
        """
        Search for all occurrences of a pattern in the text.

        Args:
            pattern: The pattern to search for

        Returns:
            List of positions where the pattern occurs
        """
        if not pattern or not self.text:
            return []

        node = self.root
        pattern_index = 0

        # Traverse the tree to find the pattern
        while pattern_index < len(pattern):
            if pattern[pattern_index] not in node.children:
                return []

            child = node.children[pattern[pattern_index]]
            edge_length = self._edge_length(child)

            # Check if the pattern matches the edge
            j = 0
            while j < edge_length and pattern_index < len(pattern):
                if pattern[pattern_index] != self.text[child.start + j]:
                    return []
                pattern_index += 1
                j += 1

            # Move to the next node
            if j == edge_length:
                node = child
            else:
                # We've reached the end of the pattern
                break

        # If we've matched the entire pattern, collect all positions
        if pattern_index == len(pattern):
            return self._collect_positions(node)

        return []

    def _collect_positions(self, node: SuffixTreeNode) -> List[int]:
        """
        Collect all positions under a node.

        Args:
            node: The node to collect positions from

        Returns:
            List of positions
        """
        positions = node.positions.copy()

        # Collect positions from all children (leaves)
        for child in node.children.values():
            positions.extend(self._collect_positions(child))

        return positions


class SuffixArray:
    """
    Suffix Array implementation.

    A suffix array is a sorted array of all suffixes of a string.
    It allows for efficient pattern matching operations with low memory usage.
    """

    def __init__(self, text: str = None):
        """
        Initialize a Suffix Array, optionally with a text.

        Args:
            text: The text to build the suffix array from
        """
        self.text = ""
        self.suffix_array = []
        self.lcp_array = []  # Longest Common Prefix array

        if text:
            self.build(text)

    def build(self, text: str) -> None:
        """
        Build the suffix array for a text.

        Args:
            text: The text to build the suffix array from
        """
        self.text = text

        # Naive O(n²log n) construction: sort all suffixes
        suffixes = [(i, text[i:]) for i in range(len(text))]
        suffixes.sort(key=lambda x: x[1])

        self.suffix_array = [pos for pos, _ in suffixes]
        self._build_lcp_array()

    def _build_lcp_array(self) -> None:
        """Build the Longest Common Prefix array."""
        n = len(self.text)
        if n == 0:
            self.lcp_array = []
            return

        # Inverse suffix array
        inv_suffix = [0] * n
        for i in range(n):
            inv_suffix[self.suffix_array[i]] = i

        # Compute LCP values
        self.lcp_array = [0] * n
        h = 0

        for i in range(n):
            if inv_suffix[i] == n - 1:
                h = 0
                continue

            j = self.suffix_array[inv_suffix[i] + 1]

            # Compute longest common prefix
            while i + h < n and j + h < n and self.text[i + h] == self.text[j + h]:
                h += 1

            self.lcp_array[inv_suffix[i]] = h

            if h > 0:
                h -= 1

    def search(self, pattern: str) -> List[int]:
        """
        Search for all occurrences of a pattern in the text.

        Args:
            pattern: The pattern to search for

        Returns:
            List of positions where the pattern occurs
        """
        if not pattern or not self.text:
            return []

        # Binary search to find the range of suffixes that start with the pattern
        left = 0
        right = len(self.suffix_array) - 1

        # Find left boundary
        while left <= right:
            mid = (left + right) // 2
            suffix = self.text[self.suffix_array[mid]:]

            if suffix >= pattern:
                right = mid - 1
            else:
                left = mid + 1

        if left >= len(self.suffix_array):
            return []

        left_boundary = left

        # Reset and find right boundary
        left = left_boundary
        right = len(self.suffix_array) - 1

        while left <= right:
            mid = (left + right) // 2
            suffix = self.text[self.suffix_array[mid]:]

            if suffix.startswith(pattern):
                left = mid + 1
            else:
                right = mid - 1

        right_boundary = right

        # If no match found
        if right_boundary < left_boundary:
            return []

        # Collect all positions
        return [self.suffix_array[i] for i in range(left_boundary, right_boundary + 1)]

    def longest_repeated_substring(self) -> Tuple[str, List[int]]:
        """
        Find the longest substring that appears at least twice in the text.

        Returns:
            Tuple of (substring, list of positions where it appears)
        """
        if not self.text or not self.lcp_array:
            return "", []

        # Find maximum LCP value
        max_lcp = 0
        max_index = 0

        for i, lcp in enumerate(self.lcp_array):
            if lcp > max_lcp:
                max_lcp = lcp
                max_index = i

        if max_lcp == 0:
            return "", []  # No repeated substring

        # Extract the substring
        substring = self.text[self.suffix_array[max_index]:self.suffix_array[max_index] + max_lcp]

        # Find all occurrences
        return substring, self.search(substring)

    def longest_common_substring(self, other_text: str) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Find the longest substring common to both texts.

        Args:
            other_text: The other text to compare with

        Returns:
            Tuple of (substring, list of (pos1, pos2) pairs where it appears)
        """
        # Concatenate texts with a separator
        separator = chr(ord(max(self.text + other_text)) + 1)
        combined = self.text + separator + other_text

        # Build suffix array for the combined text
        combined_sa = SuffixArray(combined)

        # Find longest common substring
        max_lcp = 0
        max_index = 0
        n = len(self.text)

        for i in range(1, len(combined_sa.suffix_array)):
            # Check if suffixes come from different texts
            pos1 = combined_sa.suffix_array[i]
            pos2 = combined_sa.suffix_array[i - 1]

            if (pos1 < n) != (pos2 < n):  # One from each text
                lcp = 0
                j = 0

                # Compute LCP manually
                while pos1 + j < len(combined) and pos2 + j < len(combined):
                    if combined[pos1 + j] != combined[pos2 + j]:
                        break
                    if combined[pos1 + j] == separator:
                        break
                    j += 1

                lcp = j

                if lcp > max_lcp:
                    max_lcp = lcp
                    max_index = i

        if max_lcp == 0:
            return "", []  # No common substring

        # Extract the substring
        pos = combined_sa.suffix_array[max_index]
        substring = combined[pos:pos + max_lcp]

        # Find all occurrences in both texts
        positions1 = self.search(substring)
        positions2 = [(p - n - 1) for p in combined_sa.search(substring) if p > n]

        # Create pairs of positions
        pairs = [(p1, p2) for p1 in positions1 for p2 in positions2]

        return substring, pairs


def z_algorithm(s: str) -> List[int]:
    """
    Z Algorithm for pattern matching.

    This algorithm computes the Z array, where Z[i] is the length of the longest
    substring starting at position i that is also a prefix of the string.

    Args:
        s: The input string

    Returns:
        Z array
    """
    n = len(s)
    z = [0] * n

    # Z[0] is meaningless as the entire string is a prefix of itself
    if n == 0:
        return z

    z[0] = n

    # Initialize left and right boundaries of Z-box
    l, r = 0, 0

    for i in range(1, n):
        if i > r:
            # Outside current Z-box, compute Z[i] naively
            l = r = i
            while r < n and s[r - l] == s[r]:
                r += 1
            z[i] = r - l
            r -= 1
        else:
            # Inside current Z-box, use previously computed values
            k = i - l

            # If Z[k] does not exceed the right boundary
            if z[k] < r - i + 1:
                z[i] = z[k]
            else:
                # Otherwise, we need to do a naive check
                l = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                z[i] = r - l
                r -= 1

    return z


def z_algorithm_search(text: str, pattern: str) -> List[int]:
    """
    Pattern matching using Z algorithm.

    Args:
        text: The text to search in
        pattern: The pattern to search for

    Returns:
        List of positions where the pattern occurs
    """
    # Concatenate pattern, separator and text
    separator = chr(ord(max(text + pattern)) + 1)
    combined = pattern + separator + text

    # Compute Z array
    z = z_algorithm(combined)

    # Find all occurrences
    pattern_length = len(pattern)
    positions = []

    for i in range(pattern_length + 1, len(combined)):
        if z[i] == pattern_length:
            positions.append(i - pattern_length - 1)

    return positions