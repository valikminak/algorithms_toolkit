# tree/base.py
from typing import List, Dict, Tuple, Set, Optional, Any


class TreeNode:
    """Basic tree node implementation."""

    def __init__(self, value: Any):
        """
        Initialize a tree node.

        Args:
            value: The value to store in the node
        """
        self.value = value
        self.children = []

    def add_child(self, child: 'TreeNode') -> None:
        """
        Add a child node.

        Args:
            child: The child node to add
        """
        self.children.append(child)

    def __str__(self) -> str:
        """String representation of the node."""
        return str(self.value)

    def __repr__(self) -> str:
        """Detailed string representation of the node."""
        return f"TreeNode({self.value}, {len(self.children)} children)"


class BinaryTreeNode:
    """Binary tree node implementation."""

    def __init__(self, value: Any):
        """
        Initialize a binary tree node.

        Args:
            value: The value to store in the node
        """
        self.value = value
        self.left = None
        self.right = None

    def __str__(self) -> str:
        """String representation of the node."""
        return str(self.value)

    def __repr__(self) -> str:
        """Detailed string representation of the node."""
        return f"BinaryTreeNode({self.value})"


class BinarySearchTree:
    """Binary Search Tree implementation."""

    def __init__(self):
        """Initialize an empty binary search tree."""
        self.root = None

    def insert(self, value: Any) -> None:
        """
        Insert a value into the BST.

        Args:
            value: The value to insert
        """
        if self.root is None:
            self.root = BinaryTreeNode(value)
            return

        self._insert_recursive(self.root, value)

    def _insert_recursive(self, node: BinaryTreeNode, value: Any) -> None:
        """
        Recursively insert a value into the BST.

        Args:
            node: Current node in the recursion
            value: Value to insert
        """
        if value < node.value:
            if node.left is None:
                node.left = BinaryTreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = BinaryTreeNode(value)
            else:
                self._insert_recursive(node.right, value)

    def search(self, value: Any) -> Optional[BinaryTreeNode]:
        """
        Search for a value in the BST.

        Args:
            value: The value to search for

        Returns:
            The node containing the value, or None if not found
        """
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node: Optional[BinaryTreeNode], value: Any) -> Optional[BinaryTreeNode]:
        """
        Recursively search for a value in the BST.

        Args:
            node: Current node in the recursion
            value: Value to search for

        Returns:
            The node containing the value, or None if not found
        """
        if node is None or node.value == value:
            return node

        if value < node.value:
            return self._search_recursive(node.left, value)
        return self._search_recursive(node.right, value)

    def delete(self, value: Any) -> None:
        """
        Delete a value from the BST.

        Args:
            value: The value to delete
        """
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, node: Optional[BinaryTreeNode], value: Any) -> Optional[BinaryTreeNode]:
        """
        Recursively delete a value from the BST.

        Args:
            node: Current node in the recursion
            value: Value to delete

        Returns:
            The updated node after deletion
        """
        if node is None:
            return None

        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            # Node with only one child or no child
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            # Node with two children: Get the inorder successor (smallest in right subtree)
            node.value = self._min_value(node.right)

            # Delete the inorder successor
            node.right = self._delete_recursive(node.right, node.value)

        return node

    def _min_value(self, node: BinaryTreeNode) -> Any:
        """
        Find the minimum value in a subtree.

        Args:
            node: Root of the subtree

        Returns:
            The minimum value in the subtree
        """
        current = node
        while current.left is not None:
            current = current.left
        return current.value

    def inorder_traversal(self) -> List[Any]:
        """
        Perform an inorder traversal of the BST.

        Returns:
            List of values in sorted order
        """
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node: Optional[BinaryTreeNode], result: List[Any]) -> None:
        """
        Recursively perform inorder traversal.

        Args:
            node: Current node in the recursion
            result: List to store traversal result
        """
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)


class NaryTree:
    """N-ary tree implementation."""

    def __init__(self, value: Any = None):
        """
        Initialize an N-ary tree.

        Args:
            value: Optional value for the root node
        """
        self.root = TreeNode(value) if value is not None else None

    def insert(self, parent_value: Any, value: Any) -> bool:
        """
        Insert a value as a child of a node with the given parent value.

        Args:
            parent_value: Value of the parent node
            value: Value to insert

        Returns:
            True if insertion was successful, False otherwise
        """
        if self.root is None:
            self.root = TreeNode(value)
            return True

        parent = self._find_node(self.root, parent_value)
        if parent:
            parent.add_child(TreeNode(value))
            return True
        return False

    def _find_node(self, node: TreeNode, value: Any) -> Optional[TreeNode]:
        """
        Find a node with the given value.

        Args:
            node: Current node in the search
            value: Value to find

        Returns:
            The node with the given value, or None if not found
        """
        if node.value == value:
            return node

        for child in node.children:
            result = self._find_node(child, value)
            if result:
                return result

        return None

    def traversal(self, traversal_type: str = "level") -> List[Any]:
        """
        Traverse the tree.

        Args:
            traversal_type: Type of traversal ("level", "pre", "post")

        Returns:
            List of values in the specified traversal order
        """
        if self.root is None:
            return []

        if traversal_type == "level":
            return self._level_order_traversal()
        elif traversal_type == "pre":
            result = []
            self._preorder_traversal(self.root, result)
            return result
        elif traversal_type == "post":
            result = []
            self._postorder_traversal(self.root, result)
            return result
        else:
            raise ValueError(f"Unknown traversal type: {traversal_type}")

    def _level_order_traversal(self) -> List[Any]:
        """
        Perform level-order traversal.

        Returns:
            List of values in level order
        """
        if self.root is None:
            return []

        result = []
        queue = [self.root]

        while queue:
            node = queue.pop(0)
            result.append(node.value)

            for child in node.children:
                queue.append(child)

        return result

    def _preorder_traversal(self, node: TreeNode, result: List[Any]) -> None:
        """
        Recursively perform preorder traversal.

        Args:
            node: Current node in the recursion
            result: List to store traversal result
        """
        result.append(node.value)

        for child in node.children:
            self._preorder_traversal(child, result)

    def _postorder_traversal(self, node: TreeNode, result: List[Any]) -> None:
        """
        Recursively perform postorder traversal.

        Args:
            node: Current node in the recursion
            result: List to store traversal result
        """
        for child in node.children:
            self._postorder_traversal(child, result)

        result.append(node.value)