from typing import Dict, List, Optional


class TrieNode:
    """Node in a Trie (prefix tree) data structure."""

    def __init__(self):
        """Initialize an empty trie node."""
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False

    def __str__(self) -> str:
        """String representation of a trie node."""
        return f"TrieNode(children={list(self.children.keys())}, is_end={self.is_end_of_word})"


class Trie:
    """
    Trie (prefix tree) data structure for efficient string operations.

    A trie is an ordered tree data structure used to store a dynamic set or associative
    array where the keys are usually strings. All the descendants of a node have a common
    prefix of the string associated with that node, and the root is associated with the
    empty string.
    """

    def __init__(self):
        """Initialize an empty trie with just a root node."""
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.

        Args:
            word: The word to insert
        """
        node = self.root
        for char in word:
            # If character is not in current node's children, add it
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        # Mark the end of the word
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """
        Search for a word in the trie.

        Args:
            word: The word to search for

        Returns:
            True if the word exists in the trie, False otherwise
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]

        # Word exists only if we reached the end of a word marker
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word in the trie starts with the given prefix.

        Args:
            prefix: The prefix to search for

        Returns:
            True if any word starts with the prefix, False otherwise
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]

        # If we can traverse the trie along the prefix path, return True
        return True

    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """
        Get all words in the trie that start with the given prefix.

        Args:
            prefix: The prefix to search for

        Returns:
            List of words that start with the prefix
        """
        result = []
        node = self.root

        # Traverse to the node representing the prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Use DFS to find all words from this node
        self._dfs(node, prefix, result)
        return result

    def _dfs(self, node: TrieNode, current_word: str, result: List[str]) -> None:
        """
        Helper method to perform depth-first search from a node.

        Args:
            node: Current trie node
            current_word: Word formed so far
            result: List to store found words
        """
        if node.is_end_of_word:
            result.append(current_word)

        for char, child_node in node.children.items():
            self._dfs(child_node, current_word + char, result)

    def delete(self, word: str) -> bool:
        """
        Delete a word from the trie if it exists.

        Args:
            word: The word to delete

        Returns:
            True if the word was deleted, False if it wasn't in the trie
        """
        return self._delete_helper(self.root, word, 0)

    def _delete_helper(self, node: TrieNode, word: str, index: int) -> bool:
        """
        Recursive helper method for deleting a word.

        Args:
            node: Current trie node
            word: Word to delete
            index: Current character index in the word

        Returns:
            True if the word was deleted, False otherwise
        """
        # Base case: reached the end of the word
        if index == len(word):
            # Word doesn't exist
            if not node.is_end_of_word:
                return False

            # Mark as not the end of a word
            node.is_end_of_word = False

            # Return true if node has no children, indicating it can be deleted
            return len(node.children) == 0

        char = word[index]
        if char not in node.children:
            return False

        # Recursive call for the next character
        should_delete_child = self._delete_helper(node.children[char], word, index + 1)

        # If the child should be deleted
        if should_delete_child:
            del node.children[char]

            # Return true if this node is not the end of another word and has no other children
            return not node.is_end_of_word and len(node.children) == 0

        return False

    def get_all_words(self) -> List[str]:
        """
        Get all words stored in the trie.

        Returns:
            List of all words in the trie
        """
        result = []
        self._dfs(self.root, "", result)
        return result

    def count_words(self) -> int:
        """
        Count the number of words stored in the trie.

        Returns:
            Number of words in the trie
        """
        return len(self.get_all_words())

    def __str__(self) -> str:
        """String representation of the trie."""
        words = self.get_all_words()
        return f"Trie(words={words})"


class CompressedTrieNode:
    """Node in a compressed trie (radix tree) data structure."""

    def __init__(self):
        """Initialize an empty compressed trie node."""
        self.children: Dict[str, 'CompressedTrieNode'] = {}
        self.is_end_of_word: bool = False

    def __str__(self) -> str:
        """String representation of a compressed trie node."""
        return f"CompressedTrieNode(children={list(self.children.keys())}, is_end={self.is_end_of_word})"


class CompressedTrie:
    """
    Compressed Trie (radix tree) data structure for space-efficient string storage.

    A compressed trie is a space-optimized version of a trie where each node with only
    one child is merged with its child, compressing paths that have only one possible
    traversal into a single edge labeled with a substring rather than single characters.
    """

    def __init__(self):
        """Initialize an empty compressed trie with just a root node."""
        self.root = CompressedTrieNode()

    def insert(self, word: str) -> None:
        """
        Insert a word into the compressed trie.

        Args:
            word: The word to insert
        """
        if not word:
            return

        self._insert_helper(self.root, word)

    def _insert_helper(self, node: CompressedTrieNode, word: str) -> None:
        """
        Helper method for inserting a word.

        Args:
            node: Current node
            word: Word to insert
        """
        # Try to find a matching prefix in the children
        for prefix, child in node.children.items():
            i = 0
            while i < len(prefix) and i < len(word) and prefix[i] == word[i]:
                i += 1

            # If we found a common prefix
            if i > 0:
                # Case 1: Prefix is completely matched
                if i == len(prefix):
                    if i == len(word):
                        # The word is already a prefix, just mark the end
                        child.is_end_of_word = True
                    else:
                        # Continue with the rest of the word
                        self._insert_helper(child, word[i:])
                    return

                # Case 2: Word is completely matched but prefix continues
                if i == len(word):
                    # Split the prefix
                    new_node = CompressedTrieNode()
                    new_node.is_end_of_word = True
                    new_node.children[prefix[i:]] = child

                    # Update the current node's child
                    del node.children[prefix]
                    node.children[word] = new_node
                    return

                # Case 3: Partial match of both word and prefix
                # Split the node
                new_node = CompressedTrieNode()
                new_node.children[prefix[i:]] = child

                remainder_node = CompressedTrieNode()
                remainder_node.is_end_of_word = True
                if i < len(word):
                    new_node.children[word[i:]] = remainder_node
                else:
                    new_node.is_end_of_word = True

                # Update the current node's child
                del node.children[prefix]
                node.children[prefix[:i]] = new_node
                return

        # No matching prefix found, add a new child
        new_node = CompressedTrieNode()
        new_node.is_end_of_word = True
        node.children[word] = new_node

    def search(self, word: str) -> bool:
        """
        Search for a word in the compressed trie.

        Args:
            word: The word to search for

        Returns:
            True if the word exists in the trie, False otherwise
        """
        node, remaining = self._find_node(self.root, word)
        return node is not None and remaining == "" and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word in the compressed trie starts with the given prefix.

        Args:
            prefix: The prefix to search for

        Returns:
            True if any word starts with the prefix, False otherwise
        """
        node, remaining = self._find_node(self.root, prefix)
        return node is not None and (remaining == "" or any(edge.startswith(remaining) for edge in node.children))

    def _find_node(self, node: CompressedTrieNode, string: str) -> tuple[Optional[CompressedTrieNode], str]:
        """
        Find the node that contains the string or the closest node to it.

        Args:
            node: Current node
            string: String to find

        Returns:
            Tuple of (node, remaining_string) where:
            - node is the found node, or None if not found
            - remaining_string is the part of the string not matched yet
        """
        if not string:
            return node, ""

        for prefix, child in node.children.items():
            if string.startswith(prefix):
                return self._find_node(child, string[len(prefix):])

            # Check partial match
            i = 0
            while i < len(prefix) and i < len(string) and prefix[i] == string[i]:
                i += 1

            if i > 0:
                return node, string

        return None, string

    def get_all_words(self) -> List[str]:
        """
        Get all words stored in the compressed trie.

        Returns:
            List of all words in the trie
        """
        result = []
        self._collect_words(self.root, "", result)
        return result

    def _collect_words(self, node: CompressedTrieNode, prefix: str, result: List[str]) -> None:
        """
        Helper method to collect all words from a node.

        Args:
            node: Current node
            prefix: Prefix formed so far
            result: List to store collected words
        """
        if node.is_end_of_word:
            result.append(prefix)

        for edge, child in node.children.items():
            self._collect_words(child, prefix + edge, result)