from typing import Optional
import collections
from kosmos.faw.tree.base import TreeNode, BinaryTreeNode


def tree_height(root: TreeNode) -> int:
    """
    Calculate the height of a tree.

    Args:
        root: The root node of the tree

    Returns:
        The height of the tree (0 for empty tree)
    """
    if root is None:
        return 0

    if not root.children:
        return 1

    return 1 + max(tree_height(child) for child in root.children)


def binary_tree_height(root: BinaryTreeNode) -> int:
    """
    Calculate the height of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        The height of the binary tree (0 for empty tree)
    """
    if root is None:
        return 0

    return 1 + max(binary_tree_height(root.left), binary_tree_height(root.right))


def binary_tree_size(root: BinaryTreeNode) -> int:
    """
    Calculate the size (number of nodes) of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        The number of nodes in the binary tree
    """
    if root is None:
        return 0

    return 1 + binary_tree_size(root.left) + binary_tree_size(root.right)


def binary_tree_is_balanced(root: BinaryTreeNode) -> bool:
    """
    Check if a binary tree is balanced (the height difference between left and right subtrees is at most 1).

    Args:
        root: The root node of the binary tree

    Returns:
        True if the binary tree is balanced, False otherwise
    """

    def check_height(node):
        if node is None:
            return 0

        left_height = check_height(node.left)
        if left_height == -1:
            return -1  # Left subtree is unbalanced

        right_height = check_height(node.right)
        if right_height == -1:
            return -1  # Right subtree is unbalanced

        if abs(left_height - right_height) > 1:
            return -1  # Current node is unbalanced

        return 1 + max(left_height, right_height)

    return check_height(root) != -1


def binary_tree_is_bst(root: BinaryTreeNode) -> bool:
    """
    Check if a binary tree is a valid binary search tree (BST).

    Args:
        root: The root node of the binary tree

    Returns:
        True if the binary tree is a valid BST, False otherwise
    """

    def is_valid_bst(node, min_val=float('-inf'), max_val=float('inf')):
        if node is None:
            return True

        if node.value <= min_val or node.value >= max_val:
            return False

        return (is_valid_bst(node.left, min_val, node.value) and
                is_valid_bst(node.right, node.value, max_val))

    return is_valid_bst(root)


def binary_tree_lowest_common_ancestor(root: BinaryTreeNode, p: BinaryTreeNode, q: BinaryTreeNode) -> Optional[
    BinaryTreeNode]:
    """
    Find the lowest common ancestor of two nodes in a binary tree.

    Args:
        root: The root node of the binary tree
        p, q: The two nodes to find the lowest common ancestor for

    Returns:
        The lowest common ancestor node, or None if not found
    """
    if not root or root == p or root == q:
        return root

    left = binary_tree_lowest_common_ancestor(root.left, p, q)
    right = binary_tree_lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root
    return left if left else right


def binary_tree_serialize(root: BinaryTreeNode) -> str:
    """
    Serialize a binary tree to a string.

    Args:
        root: The root node of the binary tree

    Returns:
        String representation of the binary tree
    """
    if not root:
        return "null"

    return (str(root.value) + "," +
            binary_tree_serialize(root.left) + "," +
            binary_tree_serialize(root.right))


def binary_tree_deserialize(data: str) -> Optional[BinaryTreeNode]:
    """
    Deserialize a string to a binary tree.

    Args:
        data: String representation of the binary tree

    Returns:
        The root node of the deserialized binary tree
    """

    def deserialize_helper(nodes):
        if not nodes:
            return None

        value = nodes.popleft()
        if value == "null":
            return None

        node = BinaryTreeNode(value)
        node.left = deserialize_helper(nodes)
        node.right = deserialize_helper(nodes)

        return node

    nodes = collections.deque(data.split(","))
    return deserialize_helper(nodes)


def is_complete_binary_tree(root: BinaryTreeNode) -> bool:
    """
    Check if a binary tree is a complete binary tree.

    A complete binary tree is a binary tree in which all levels are filled, except possibly the last level,
    which is filled from left to right.

    Args:
        root: The root node of the binary tree

    Returns:
        True if the binary tree is complete, False otherwise
    """
    if not root:
        return True

    queue = collections.deque([root])
    end_reached = False

    while queue:
        node = queue.popleft()

        # If we've reached an empty node, all remaining nodes should be empty
        if node is None:
            end_reached = True
        else:
            # If we've already reached an empty node and now we see a non-empty node,
            # the tree is not complete
            if end_reached:
                return False

            # Add both left and right children to the queue (even if they're None)
            queue.append(node.left)
            queue.append(node.right)

    return True


def is_full_binary_tree(root: BinaryTreeNode) -> bool:
    """
    Check if a binary tree is a full binary tree.

    A full binary tree is a binary tree in which every node has either 0 or 2 children.

    Args:
        root: The root node of the binary tree

    Returns:
        True if the binary tree is full, False otherwise
    """
    if not root:
        return True

    # If a node has no children, it's a leaf node
    if not root.left and not root.right:
        return True

    # If a node has both children, check both subtrees
    if root.left and root.right:
        return is_full_binary_tree(root.left) and is_full_binary_tree(root.right)

    # If a node has only one child, it's not a full binary tree
    return False


def is_perfect_binary_tree(root: BinaryTreeNode) -> bool:
    """
    Check if a binary tree is a perfect binary tree.

    A perfect binary tree is a binary tree in which all interior nodes have two children
    and all leaves are at the same level.

    Args:
        root: The root node of the binary tree

    Returns:
        True if the binary tree is perfect, False otherwise
    """

    # First, get the height of the tree
    def get_height(node):
        if not node:
            return 0
        return 1 + get_height(node.left)

    height = get_height(root)

    # Then, check if the tree is perfect
    def is_perfect(node, level, height):
        # An empty tree is perfect
        if not node:
            return True

        # If it's a leaf node, it must be at the last level
        if not node.left and not node.right:
            return level == height

        # If it's an internal node, it must have both children
        if not node.left or not node.right:
            return False

        # Check both subtrees
        return is_perfect(node.left, level + 1, height) and is_perfect(node.right, level + 1, height)

    return is_perfect(root, 1, height)


def count_leaves(root: BinaryTreeNode) -> int:
    """
    Count the number of leaf nodes in a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        The number of leaf nodes
    """
    if not root:
        return 0

    # If it's a leaf node
    if not root.left and not root.right:
        return 1

    # Count leaves in both subtrees
    return count_leaves(root.left) + count_leaves(root.right)


def tree_diameter(root: BinaryTreeNode) -> int:
    """
    Calculate the diameter of a binary tree.

    The diameter of a binary tree is the length of the longest path between any two nodes.
    This path may or may not pass through the root.

    Args:
        root: The root node of the binary tree

    Returns:
        The diameter of the tree
    """
    diameter = 0

    def height(node):
        nonlocal diameter

        if not node:
            return 0

        left_height = height(node.left)
        right_height = height(node.right)

        # Update diameter if path through this node is longer
        diameter = max(diameter, left_height + right_height)

        # Return the height of the subtree rooted at this node
        return 1 + max(left_height, right_height)

    height(root)
    return diameter


def mirror_tree(root: BinaryTreeNode) -> BinaryTreeNode:
    """
    Mirror a binary tree (swap left and right subtrees).

    Args:
        root: The root node of the binary tree

    Returns:
        The root node of the mirrored tree
    """
    if not root:
        return None

    # Swap the subtrees
    root.left, root.right = root.right, root.left

    # Recursively mirror the subtrees
    mirror_tree(root.left)
    mirror_tree(root.right)

    return root


def are_trees_identical(root1: BinaryTreeNode, root2: BinaryTreeNode) -> bool:
    """
    Check if two binary trees are identical.

    Args:
        root1: The root node of the first binary tree
        root2: The root node of the second binary tree

    Returns:
        True if the trees are identical, False otherwise
    """
    # If both trees are empty, they're identical
    if not root1 and not root2:
        return True

    # If one tree is empty but the other isn't, they're not identical
    if not root1 or not root2:
        return False

    # Check if current nodes have the same value and both subtrees are identical
    return (root1.value == root2.value and
            are_trees_identical(root1.left, root2.left) and
            are_trees_identical(root1.right, root2.right))


def print_tree(root: BinaryTreeNode, level: int = 0) -> None:
    """
    Print a binary tree in a readable format.

    Args:
        root: The root node of the binary tree
        level: Current level in the tree (for indentation)
    """
    if not root:
        return

    # Print right subtree
    print_tree(root.right, level + 1)

    # Print current node
    print("    " * level + str(root.value))

    # Print left subtree
    print_tree(root.left, level + 1)