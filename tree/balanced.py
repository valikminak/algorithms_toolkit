from enum import Enum
from typing import Any, Optional, List, Generator, Tuple


class Color(Enum):
    """Color enum for Red-Black Tree nodes."""
    RED = 1
    BLACK = 2


class RedBlackNode:
    """Node in a Red-Black Tree."""

    def __init__(self, key: Any, value: Any = None, color: Color = Color.RED):
        """
        Initialize a Red-Black Tree node.

        Args:
            key: Key for ordering
            value: Associated value (optional)
            color: Node color (RED by default)
        """
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.color = color

    def __str__(self) -> str:
        """String representation of the node."""
        color_str = "RED" if self.color == Color.RED else "BLACK"
        return f"({self.key}:{color_str})"


class RedBlackTree:
    """
    Red-Black Tree implementation.

    A Red-Black Tree is a self-balancing binary search tree where each node has a color
    (red or black) that satisfies the following properties:
    1. Every node is either red or black
    2. The root is black
    3. Every leaf (NIL) is black
    4. If a node is red, then both its children are black
    5. For each node, all simple paths from the node to descendant leaves contain the
       same number of black nodes (black-height)
    """

    def __init__(self):
        """Initialize an empty Red-Black Tree."""
        self.NIL = RedBlackNode(None, None, Color.BLACK)  # Sentinel node
        self.root = self.NIL

    def search(self, key: Any) -> Optional[Any]:
        """
        Search for a key in the tree.

        Args:
            key: Key to search for

        Returns:
            Value associated with the key, or None if not found
        """
        node = self._search_node(key)
        return node.value if node != self.NIL else None

    def _search_node(self, key: Any) -> RedBlackNode:
        """
        Search for a node with the given key.

        Args:
            key: Key to search for

        Returns:
            The node if found, otherwise NIL
        """
        current = self.root
        while current != self.NIL:
            if key == current.key:
                return current
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        return self.NIL

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Insert a key-value pair into the tree.

        Args:
            key: Key for ordering
            value: Associated value
        """
        # Create new node
        new_node = RedBlackNode(key, value)
        new_node.left = self.NIL
        new_node.right = self.NIL

        # Regular BST insertion
        parent = self.NIL
        current = self.root

        while current != self.NIL:
            parent = current
            if new_node.key < current.key:
                current = current.left
            else:
                current = current.right

        new_node.parent = parent

        if parent == self.NIL:
            # Tree was empty
            self.root = new_node
        elif new_node.key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        # Fix Red-Black properties
        self._fix_insert(new_node)

    def _fix_insert(self, node: RedBlackNode) -> None:
        """
        Fix Red-Black Tree properties after insertion.

        Args:
            node: The newly inserted node
        """
        # Fix the tree
        while node != self.root and node.parent.color == Color.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right

                if uncle.color == Color.RED:
                    # Case 1: Uncle is red
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Uncle is black and node is a right child
                        node = node.parent
                        self._left_rotate(node)

                    # Case 3: Uncle is black and node is a left child
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left

                if uncle.color == Color.RED:
                    # Case 1: Uncle is red
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        # Case 2: Uncle is black and node is a left child
                        node = node.parent
                        self._right_rotate(node)

                    # Case 3: Uncle is black and node is a right child
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._left_rotate(node.parent.parent)

        # Ensure root is black
        self.root.color = Color.BLACK

    def _left_rotate(self, x: RedBlackNode) -> None:
        """
        Perform a left rotation.

        Args:
            x: The node around which to rotate
        """
        y = x.right

        # Turn y's left subtree into x's right subtree
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x

        # Link y's parent to x's parent
        y.parent = x.parent
        if x.parent == self.NIL:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        # Put x on y's left
        y.left = x
        x.parent = y

    def _right_rotate(self, y: RedBlackNode) -> None:
        """
        Perform a right rotation.

        Args:
            y: The node around which to rotate
        """
        x = y.left

        # Turn x's right subtree into y's left subtree
        y.left = x.right
        if x.right != self.NIL:
            x.right.parent = y

        # Link x's parent to y's parent
        x.parent = y.parent
        if y.parent == self.NIL:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x

        # Put y on x's right
        x.right = y
        y.parent = x

    def delete(self, key: Any) -> bool:
        """
        Delete a node with the given key.

        Args:
            key: Key to delete

        Returns:
            True if the key was found and deleted, False otherwise
        """
        # Find the node to delete
        z = self._search_node(key)
        if z == self.NIL:
            return False

        # Remember the color and find replacement
        y = z
        y_original_color = y.color

        if z.left == self.NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right == self.NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            # Find successor
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right

            if y.parent == z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        # Fix Red-Black properties if needed
        if y_original_color == Color.BLACK:
            self._fix_delete(x)

        return True

    def _transplant(self, u: RedBlackNode, v: RedBlackNode) -> None:
        """
        Replace subtree rooted at u with subtree rooted at v.

        Args:
            u: The node to be replaced
            v: The replacement node
        """
        if u.parent == self.NIL:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, node: RedBlackNode) -> RedBlackNode:
        """
        Find the minimum key in the subtree rooted at node.

        Args:
            node: Root of the subtree

        Returns:
            Node with minimum key
        """
        current = node
        while current.left != self.NIL:
            current = current.left
        return current

    def _fix_delete(self, x: RedBlackNode) -> None:
        """
        Fix Red-Black Tree properties after deletion.

        Args:
            x: The node that replaced the deleted node
        """
        while x != self.root and x.color == Color.BLACK:
            if x == x.parent.left:
                w = x.parent.right

                if w.color == Color.RED:
                    # Case 1: x's sibling w is red
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._left_rotate(x.parent)
                    w = x.parent.right

                if w.left.color == Color.BLACK and w.right.color == Color.BLACK:
                    # Case 2: Both of w's children are black
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.right.color == Color.BLACK:
                        # Case 3: w's right child is black
                        w.left.color = Color.BLACK
                        w.color = Color.RED
                        self._right_rotate(w)
                        w = x.parent.right

                    # Case 4: w's right child is red
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.right.color = Color.BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                w = x.parent.left

                if w.color == Color.RED:
                    # Case 1: x's sibling w is red
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._right_rotate(x.parent)
                    w = x.parent.left

                if w.right.color == Color.BLACK and w.left.color == Color.BLACK:
                    # Case 2: Both of w's children are black
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.left.color == Color.BLACK:
                        # Case 3: w's left child is black
                        w.right.color = Color.BLACK
                        w.color = Color.RED
                        self._left_rotate(w)
                        w = x.parent.left

                    # Case 4: w's left child is red
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.left.color = Color.BLACK
                    self._right_rotate(x.parent)
                    x = self.root

        x.color = Color.BLACK

    def inorder_traversal(self) -> Generator[Tuple[Any, Any], None, None]:
        """
        Perform an inorder traversal of the tree.

        Yields:
            Tuples of (key, value) in sorted order
        """

        def _inorder(node):
            if node != self.NIL:
                yield from _inorder(node.left)
                yield (node.key, node.value)
                yield from _inorder(node.right)

        yield from _inorder(self.root)

    def is_valid(self) -> bool:
        """
        Check if the tree satisfies all Red-Black properties.

        Returns:
            True if the tree is a valid Red-Black Tree, False otherwise
        """
        if self.root == self.NIL:
            return True

        # Property 1 and 2 (Root is black) are enforced in the insert method

        # Check property 4 (Red nodes have black children)
        if not self._check_red_property(self.root):
            return False

        # Check property 5 (Black height is the same for all paths)
        black_height = -1
        return self._check_black_height(self.root, 0, black_height) >= 0

    def _check_red_property(self, node: RedBlackNode) -> bool:
        """
        Check if the Red-Black property 4 is satisfied.

        Args:
            node: Root of the subtree to check

        Returns:
            True if property 4 is satisfied, False otherwise
        """
        if node == self.NIL:
            return True

        if node.color == Color.RED:
            if (node.left != self.NIL and node.left.color == Color.RED) or \
                    (node.right != self.NIL and node.right.color == Color.RED):
                return False

        return self._check_red_property(node.left) and self._check_red_property(node.right)

    def _check_black_height(self, node: RedBlackNode, black_count: int, expected: int) -> int:
        """
        Check if the Red-Black property 5 is satisfied and compute black height.

        Args:
            node: Root of the subtree to check
            black_count: Number of black nodes seen so far
            expected: Expected black height (-1 if not yet determined)

        Returns:
            Black height if all paths have the same black height, -1 otherwise
        """
        if node == self.NIL:
            if expected == -1:
                return black_count
            return black_count if black_count == expected else -1

        # Count black nodes
        if node.color == Color.BLACK:
            black_count += 1

        left_height = self._check_black_height(node.left, black_count, expected)
        if left_height == -1:
            return -1

        return self._check_black_height(node.right, black_count, left_height)

    def __str__(self) -> str:
        """String representation of the tree."""
        if self.root == self.NIL:
            return "Empty"

        lines = []
        self._print_tree(self.root, 0, lines)
        return "\n".join(lines)

    def _print_tree(self, node: RedBlackNode, indent: int, lines: List[str]) -> None:
        """
        Helper function for string representation.

        Args:
            node: Current node
            indent: Current indentation level
            lines: List to store output lines
        """
        if node != self.NIL:
            self._print_tree(node.right, indent + 4, lines)

            lines.append(" " * indent + str(node))

            self._print_tree(node.left, indent + 4, lines)


class AVLNode:
    """Node in an AVL Tree."""

    def __init__(self, key: Any, value: Any = None):
        """
        Initialize an AVL Tree node.

        Args:
            key: Key for ordering
            value: Associated value (optional)
        """
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.height = 1  # Height of node (1 for leaf)

    def __str__(self) -> str:
        """String representation of the node."""
        return f"({self.key}, h={self.height})"


class AVLTree:
    """
    AVL Tree implementation.

    An AVL Tree is a self-balancing binary search tree where the heights of the
    two child subtrees of any node differ by at most one.
    """

    def __init__(self):
        """Initialize an empty AVL Tree."""
        self.root = None

    def height(self, node: Optional[AVLNode]) -> int:
        """
        Get the height of a node.

        Args:
            node: The node to get height of

        Returns:
            The height of the node, or 0 if node is None
        """
        return node.height if node else 0

    def balance_factor(self, node: AVLNode) -> int:
        """
        Calculate the balance factor of a node.

        Args:
            node: The node to calculate balance factor for

        Returns:
            The balance factor (left height - right height)
        """
        return self.height(node.left) - self.height(node.right)

    def search(self, key: Any) -> Optional[Any]:
        """
        Search for a key in the tree.

        Args:
            key: Key to search for

        Returns:
            Value associated with the key, or None if not found
        """
        node = self._search_node(self.root, key)
        return node.value if node else None

    def _search_node(self, node: Optional[AVLNode], key: Any) -> Optional[AVLNode]:
        """
        Search for a node with the given key.

        Args:
            node: Root of the subtree to search in
            key: Key to search for

        Returns:
            The node if found, otherwise None
        """
        if not node or key == node.key:
            return node

        if key < node.key:
            return self._search_node(node.left, key)
        return self._search_node(node.right, key)

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Insert a key-value pair into the tree.

        Args:
            key: Key for ordering
            value: Associated value
        """
        self.root = self._insert(self.root, key, value)

    def _insert(self, node: Optional[AVLNode], key: Any, value: Any) -> AVLNode:
        """
        Insert a key-value pair into the subtree rooted at node.

        Args:
            node: Root of the subtree
            key: Key for ordering
            value: Associated value

        Returns:
            The new root of the balanced subtree
        """
        # Standard BST insertion
        if not node:
            return AVLNode(key, value)

        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            # Update value if key already exists
            node.value = value
            return node

        # Update height
        node.height = 1 + max(self.height(node.left), self.height(node.right))

        # Get balance factor
        balance = self.balance_factor(node)

        # Left-Left Case
        if balance > 1 and key < node.left.key:
            return self._right_rotate(node)

        # Right-Right Case
        if balance < -1 and key > node.right.key:
            return self._left_rotate(node)

        # Left-Right Case
        if balance > 1 and key > node.left.key:
            node.left = self._left_rotate(node.left)
            return self._right_rotate(node)

        # Right-Left Case
        if balance < -1 and key < node.right.key:
            node.right = self._right_rotate(node.right)
            return self._left_rotate(node)

        return node

    def _left_rotate(self, z: AVLNode) -> AVLNode:
        """
        Perform a left rotation.

        Args:
            z: The node around which to rotate

        Returns:
            The new root of the rotated subtree
        """
        y = z.right
        T2 = y.left

        # Perform rotation
        y.left = z
        z.right = T2

        # Update heights
        z.height = 1 + max(self.height(z.left), self.height(z.right))
        y.height = 1 + max(self.height(y.left), self.height(y.right))

        return y

    def _right_rotate(self, z: AVLNode) -> AVLNode:
        """
        Perform a right rotation.

        Args:
            z: The node around which to rotate

        Returns:
            The new root of the rotated subtree
        """
        y = z.left
        T3 = y.right

        # Perform rotation
        y.right = z
        z.left = T3

        # Update heights
        z.height = 1 + max(self.height(z.left), self.height(z.right))
        y.height = 1 + max(self.height(y.left), self.height(y.right))

        return y

    def delete(self, key: Any) -> bool:
        """
        Delete a node with the given key.

        Args:
            key: Key to delete

        Returns:
            True if the key was found and deleted, False otherwise
        """
        if not self._search_node(self.root, key):
            return False

        self.root = self._delete(self.root, key)
        return True

    def _delete(self, node: Optional[AVLNode], key: Any) -> Optional[AVLNode]:
        """
        Delete a node with the given key from the subtree rooted at node.

        Args:
            node: Root of the subtree
            key: Key to delete

        Returns:
            The new root of the balanced subtree
        """
        if not node:
            return None

        # Standard BST delete
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # Node to be deleted found

            # Node with only one child or no child
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            # Node with two children
            # Get the inorder successor (smallest in right subtree)
            successor = self._min_value_node(node.right)

            # Copy the successor's data to this node
            node.key = successor.key
            node.value = successor.value

            # Delete the successor
            node.right = self._delete(node.right, successor.key)

        # If tree had only one node
        if not node:
            return None

        # Update height
        node.height = 1 + max(self.height(node.left), self.height(node.right))

        # Get balance factor
        balance = self.balance_factor(node)

        # Left-Left Case
        if balance > 1 and self.balance_factor(node.left) >= 0:
            return self._right_rotate(node)

        # Left-Right Case
        if balance > 1 and self.balance_factor(node.left) < 0:
            node.left = self._left_rotate(node.left)
            return self._right_rotate(node)

        # Right-Right Case
        if balance < -1 and self.balance_factor(node.right) <= 0:
            return self._left_rotate(node)

        # Right-Left Case
        if balance < -1 and self.balance_factor(node.right) > 0:
            node.right = self._right_rotate(node.right)
            return self._left_rotate(node)

        return node

    def _min_value_node(self, node: AVLNode) -> AVLNode:
        """
        Find the node with minimum key in the subtree.

        Args:
            node: Root of the subtree

        Returns:
            Node with minimum key
        """
        current = node
        while current.left:
            current = current.left
        return current

    def inorder_traversal(self) -> Generator[Tuple[Any, Any], None, None]:
        """
        Perform an inorder traversal of the tree.

        Yields:
            Tuples of (key, value) in sorted order
        """

        def _inorder(node):
            if node:
                yield from _inorder(node.left)
                yield (node.key, node.value)
                yield from _inorder(node.right)

        yield from _inorder(self.root)

    def is_balanced(self) -> bool:
        """
        Check if the tree satisfies the AVL balance property.

        Returns:
            True if the tree is balanced, False otherwise
        """
        return self._is_balanced(self.root)

    def _is_balanced(self, node: Optional[AVLNode]) -> bool:
        """
        Check if the subtree rooted at node is balanced.

        Args:
            node: Root of the subtree

        Returns:
            True if the subtree is balanced, False otherwise
        """
        if not node:
            return True

        balance = self.balance_factor(node)

        if abs(balance) > 1:
            return False

        return self._is_balanced(node.left) and self._is_balanced(node.right)

    def __str__(self) -> str:
        """String representation of the tree."""
        if not self.root:
            return "Empty"

        lines = []
        self._print_tree(self.root, 0, lines)
        return "\n".join(lines)

    def _print_tree(self, node: AVLNode, indent: int, lines: List[str]) -> None:
        """
        Helper function for string representation.

        Args:
            node: Current node
            indent: Current indentation level
            lines: List to store output lines
        """
        if node:
            self._print_tree(node.right, indent + 4, lines)

            lines.append(" " * indent + str(node))

            self._print_tree(node.left, indent + 4, lines)