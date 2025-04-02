from typing import List, Dict, Set, Any, Optional, Tuple
import collections


class AhoCorasick:
    """
    Aho-Corasick algorithm implementation for efficient multiple pattern matching.

    This algorithm can find all occurrences of a set of patterns in a text
    in linear time with respect to the length of the text plus the total
    length of the patterns.
    """

    def __init__(self):
        """Initialize an empty Aho-Corasick automaton."""
        # Root node of the trie
        self.root = {}

        # Output function: maps nodes to the patterns that end at them
        self.outputs = {}

        # Failure function: where to go when a match fails
        self.failures = {}

        # Whether the automaton is built
        self.built = False

    def add_pattern(self, pattern: str, pattern_id: Any = None) -> None:
        """
        Add a pattern to the automaton.

        Args:
            pattern: The pattern to add
            pattern_id: Optional identifier for the pattern
        """
        if self.built:
            raise ValueError("Cannot add patterns after the automaton is built")

        if pattern_id is None:
            pattern_id = pattern

        # Start at the root node
        node = self.root

        # Add the pattern to the trie
        for char in pattern:
            if char not in node:
                node[char] = {}
            node = node[char]

        # Mark the end of the pattern
        if id(node) not in self.outputs:
            self.outputs[id(node)] = []
        self.outputs[id(node)].append((pattern, pattern_id))

    def add_patterns(self, patterns: List[str]) -> None:
        """
        Add multiple patterns to the automaton.

        Args:
            patterns: List of patterns to add
        """
        for pattern in patterns:
            self.add_pattern(pattern)

    def build(self) -> None:
        """
        Build the automaton by computing failure functions.

        This must be called after adding all patterns and before searching.
        """
        # Initialize failure function for the root
        self.failures[id(self.root)] = self.root

        # BFS to compute failure function for all nodes
        queue = collections.deque()

        # Add children of root to queue
        for char, child in self.root.items():
            self.failures[id(child)] = self.root
            queue.append((child, char))

        # Process the queue
        while queue:
            node, char = queue.popleft()

            # Add child nodes to queue
            for c, child in node.items():
                queue.append((child, c))

                # Start from the failure node of the current node
                failure = self.failures[id(node)]

                # Find the failure node for the child
                while failure is not self.root and c not in failure:
                    failure = self.failures[id(failure)]

                if c in failure:
                    failure = failure[c]

                self.failures[id(child)] = failure

                # Add outputs of failure node to current node
                if id(failure) in self.outputs:
                    if id(child) not in self.outputs:
                        self.outputs[id(child)] = []
                    self.outputs[id(child)].extend(self.outputs[id(failure)])

        self.built = True

    def search(self, text: str) -> List[Tuple[int, str, Any]]:
        """
        Search for all patterns in the text.

        Args:
            text: The text to search in

        Returns:
            List of tuples (position, pattern, pattern_id) for each match
        """
        if not self.built:
            raise ValueError("Automaton must be built before searching")

        results = []
        node = self.root

        for i, char in enumerate(text):
            # Find the next node (transition)
            while node is not self.root and char not in node:
                node = self.failures[id(node)]

            if char in node:
                node = node[char]

            # Check for pattern matches at this node
            if id(node) in self.outputs:
                for pattern, pattern_id in self.outputs[id(node)]:
                    # Pattern ends at position i, so it starts at i - len(pattern) + 1
                    start_pos = i - len(pattern) + 1
                    results.append((start_pos, pattern, pattern_id))

        return results

    def search_all(self, text: str) -> Dict[str, List[int]]:
        """
        Search for all patterns in the text and return positions grouped by pattern.

        Args:
            text: The text to search in

        Returns:
            Dictionary mapping each pattern to a list of its starting positions
        """
        matches = {}

        for pos, pattern, pattern_id in self.search(text):
            if pattern not in matches:
                matches[pattern] = []
            matches[pattern].append(pos)

        return matches

    def search_first(self, text: str) -> Optional[Tuple[int, str, Any]]:
        """
        Find the first occurrence of any pattern in the text.

        Args:
            text: The text to search in

        Returns:
            Tuple (position, pattern, pattern_id) for the first match, or None if no match
        """
        if not self.built:
            raise ValueError("Automaton must be built before searching")

        node = self.root

        for i, char in enumerate(text):
            # Find the next node (transition)
            while node is not self.root and char not in node:
                node = self.failures[id(node)]

            if char in node:
                node = node[char]

            # Check for pattern matches at this node
            if id(node) in self.outputs:
                pattern, pattern_id = self.outputs[id(node)][0]
                start_pos = i - len(pattern) + 1
                return (start_pos, pattern, pattern_id)

        return None


def aho_corasick_search(text: str, patterns: List[str]) -> Dict[str, List[int]]:
    """
    Simplified interface for Aho-Corasick multiple pattern matching.

    Args:
        text: The text to search in
        patterns: List of patterns to search for

    Returns:
        Dictionary mapping each pattern to a list of its starting positions
    """
    ac = AhoCorasick()
    for pattern in patterns:
        ac.add_pattern(pattern)
    ac.build()
    return ac.search_all(text)