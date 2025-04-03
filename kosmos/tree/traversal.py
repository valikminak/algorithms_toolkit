from typing import List, Any
import collections
from kosmos.tree.base import BinaryTreeNode


def binary_tree_inorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform inorder traversal of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in inorder traversal order
    """
    result = []

    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.value)
            inorder(node.right)

    inorder(root)
    return result


def binary_tree_preorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform preorder traversal of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in preorder traversal order
    """
    result = []

    def preorder(node):
        if node:
            result.append(node.value)
            preorder(node.left)
            preorder(node.right)

    preorder(root)
    return result


def binary_tree_postorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform postorder traversal of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in postorder traversal order
    """
    result = []

    def postorder(node):
        if node:
            postorder(node.left)
            postorder(node.right)
            result.append(node.value)

    postorder(root)
    return result


def binary_tree_levelorder_traversal(root: BinaryTreeNode) -> List[List[Any]]:
    """
    Perform level-order traversal of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        List of lists, where each inner list contains node values at the same level
    """
    if not root:
        return []

    result = []
    queue = collections.deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.value)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result


def morris_inorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform inorder traversal of a binary tree using Morris Traversal.

    Morris Traversal uses threading to avoid using stack or recursion.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in inorder traversal order
    """
    result = []
    current = root

    while current:
        # If left is None, visit the node and go to right
        if not current.left:
            result.append(current.value)
            current = current.right
        else:
            # Find the inorder predecessor of current
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right

            # If right is None, go left after establishing link from predecessor to current
            if not predecessor.right:
                predecessor.right = current
                current = current.left
            else:
                # Revert the changes made in the if part to restore the original tree
                predecessor.right = None
                result.append(current.value)
                current = current.right

    return result


def morris_preorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform preorder traversal of a binary tree using Morris Traversal.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in preorder traversal order
    """
    result = []
    current = root

    while current:
        # If left is None, visit the node and go to right
        if not current.left:
            result.append(current.value)
            current = current.right
        else:
            # Find the inorder predecessor of current
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right

            # If right is None, visit current and go left after establishing link
            if not predecessor.right:
                result.append(current.value)  # Visit current node before going to left
                predecessor.right = current
                current = current.left
            else:
                # Revert the changes made in the if part to restore the original tree
                predecessor.right = None
                current = current.right

    return result


def iterative_inorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform inorder traversal of a binary tree iteratively.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in inorder traversal order
    """
    result = []
    stack = []
    current = root

    while current or stack:
        # Reach the leftmost node of the current node
        while current:
            stack.append(current)
            current = current.left

        # Current is now None, pop the next node from the stack
        current = stack.pop()
        result.append(current.value)

        # Move to the right for next iteration
        current = current.right

    return result


def iterative_preorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform preorder traversal of a binary tree iteratively.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in preorder traversal order
    """
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.value)

        # Push right child first so that left child is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result


def iterative_postorder_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform postorder traversal of a binary tree iteratively.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in postorder traversal order
    """
    if not root:
        return []

    result = []
    stack = [(root, False)]  # (node, visited_right)

    while stack:
        node, visited_right = stack.pop()

        if visited_right:
            result.append(node.value)
        else:
            # Push the node again with visited_right flag
            stack.append((node, True))

            # Push right child first, then left child
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))

    return result


def zigzag_level_order_traversal(root: BinaryTreeNode) -> List[List[Any]]:
    """
    Perform zigzag level order traversal of a binary tree.

    This traversal goes from left to right for one level, then right to left for the next.

    Args:
        root: The root node of the binary tree

    Returns:
        List of lists, where each inner list contains node values at the same level, in zigzag order
    """
    if not root:
        return []

    result = []
    queue = collections.deque([root])
    left_to_right = True

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.value)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        # Reverse level if going right to left
        if not left_to_right:
            level.reverse()

        result.append(level)
        left_to_right = not left_to_right

    return result


def boundary_traversal(root: BinaryTreeNode) -> List[Any]:
    """
    Perform boundary traversal of a binary tree.

    Boundary traversal includes: root, left boundary (excluding leaves),
    leaves from left to right, and right boundary (excluding leaves) in reverse.

    Args:
        root: The root node of the binary tree

    Returns:
        List of node values in boundary traversal order
    """
    if not root:
        return []

    result = [root.value]

    def left_boundary(node):
        if not node or (not node.left and not node.right):
            return

        result.append(node.value)

        if node.left:
            left_boundary(node.left)
        else:
            left_boundary(node.right)

    def right_boundary(node):
        if not node or (not node.left and not node.right):
            return

        if node.right:
            right_boundary(node.right)
        else:
            right_boundary(node.left)

        result.append(node.value)  # Add after recursion for reverse order

    def leaves(node):
        if not node:
            return

        if not node.left and not node.right:
            result.append(node.value)
            return

        leaves(node.left)
        leaves(node.right)

    # Skip root for all three steps (already added)
    if root.left:
        left_boundary(root.left)

    # Only add leaves if it's not already the root
    if root.left or root.right:
        leaves(root)

    # Skip right subtree's root/leftmost leaf (already added)
    if root.right:
        right_boundary(root.right)

    return result


def vertical_order_traversal(root: BinaryTreeNode) -> List[List[Any]]:
    """
    Perform vertical order traversal of a binary tree.

    Args:
        root: The root node of the binary tree

    Returns:
        List of lists, where each inner list contains node values in the same vertical line
    """
    if not root:
        return []

    # Map of horizontal distance -> list of values at that distance
    column_table = collections.defaultdict(list)

    # Queue with (node, column) pairs
    queue = collections.deque([(root, 0)])

    # Keep track of min and max column to maintain order
    min_column = max_column = 0

    while queue:
        node, column = queue.popleft()

        column_table[column].append(node.value)

        min_column = min(min_column, column)
        max_column = max(max_column, column)

        if node.left:
            queue.append((node.left, column - 1))

        if node.right:
            queue.append((node.right, column + 1))

    # Return values from leftmost to rightmost column
    return [column_table[i] for i in range(min_column, max_column + 1)]