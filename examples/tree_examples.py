from tree.base import BinaryTreeNode
from tree.traversal import (
    binary_tree_inorder_traversal, binary_tree_preorder_traversal,
    binary_tree_postorder_traversal, binary_tree_levelorder_traversal
)
from tree.properties import (
    binary_tree_height, binary_tree_is_balanced, binary_tree_is_bst,
    binary_tree_serialize, binary_tree_deserialize
)
from tree.balanced import AVLTree, RedBlackTree
from tree.segment_tree import create_sum_segment_tree
from tree.fenwick_tree import FenwickTree
from tree.trie import Trie
from tree.lru_cache import LRUCache


def create_sample_binary_tree():
    """Create a sample binary tree for demonstration."""
    root = BinaryTreeNode(1)
    root.left = BinaryTreeNode(2)
    root.right = BinaryTreeNode(3)
    root.left.left = BinaryTreeNode(4)
    root.left.right = BinaryTreeNode(5)
    root.right.left = BinaryTreeNode(6)
    root.right.right = BinaryTreeNode(7)

    return root


# examples/tree_examples.py (continued)
def traversal_example():
    """Example usage of tree traversal algorithms."""
    root = create_sample_binary_tree()

    print("Binary Tree Traversals:")
    print(f"Inorder: {binary_tree_inorder_traversal(root)}")
    print(f"Preorder: {binary_tree_preorder_traversal(root)}")
    print(f"Postorder: {binary_tree_postorder_traversal(root)}")
    print(f"Level-order: {binary_tree_levelorder_traversal(root)}")


def properties_example():
    """Example usage of tree property functions."""
    root = create_sample_binary_tree()

    print("Binary Tree Properties:")
    print(f"Height: {binary_tree_height(root)}")
    print(f"Is Balanced: {binary_tree_is_balanced(root)}")
    print(f"Is BST: {binary_tree_is_bst(root)}")

    # Create a proper BST
    bst_root = BinaryTreeNode(4)
    bst_root.left = BinaryTreeNode(2)
    bst_root.right = BinaryTreeNode(6)
    bst_root.left.left = BinaryTreeNode(1)
    bst_root.left.right = BinaryTreeNode(3)
    bst_root.right.left = BinaryTreeNode(5)
    bst_root.right.right = BinaryTreeNode(7)

    print(f"BST is valid BST: {binary_tree_is_bst(bst_root)}")

    # Serialize and deserialize
    serialized = binary_tree_serialize(root)
    print(f"Serialized tree: {serialized}")
    deserialized = binary_tree_deserialize(serialized)
    print(f"Deserialized tree (inorder): {binary_tree_inorder_traversal(deserialized)}")


def balanced_tree_example():
    """Example usage of balanced trees."""
    # AVL Tree
    avl = AVLTree()
    for i in [9, 5, 10, 0, 6, 11, -1, 1, 2]:
        avl.insert(i)

    print("AVL Tree:")
    print(f"Inorder traversal: {[k for k, v in avl.inorder_traversal()]}")
    print(f"Is balanced: {avl.is_balanced()}")

    # Red-Black Tree
    rb = RedBlackTree()
    for i in [7, 3, 18, 10, 22, 8, 11, 26]:
        rb.insert(i)

    print("\nRed-Black Tree:")
    print(f"Inorder traversal: {[k for k, v in rb.inorder_traversal()]}")


def segment_tree_example():
    """Example usage of segment tree."""
    arr = [1, 3, 5, 7, 9, 11]

    # Create a segment tree for sum queries
    st = create_sum_segment_tree(arr)

    print("Segment Tree for Sum Queries:")
    print(f"Sum of range [1, 3]: {st.query(1, 3)}")
    print(f"Sum of range [0, 5]: {st.query(0, 5)}")

    # Update a value
    st.update(2, 10)
    print("After updating index 2 to 10:")
    print(f"Sum of range [1, 3]: {st.query(1, 3)}")


def fenwick_tree_example():
    """Example usage of Fenwick Tree (Binary Indexed Tree)."""
    arr = [2, 1, 1, 3, 2, 3, 4, 5, 6, 7, 8, 9]

    # Create a Fenwick Tree
    ft = FenwickTree.from_array(arr)

    print("Fenwick Tree (Binary Indexed Tree):")
    print(f"Sum of range [0, 5]: {ft.prefix_sum(5)}")
    print(f"Sum of range [3, 7]: {ft.range_sum(3, 7)}")

    # Update a value
    ft.update(2, 5)  # Add 5 to index 2
    print("After updating index 2:")
    print(f"Sum of range [0, 5]: {ft.prefix_sum(5)}")


def trie_example():
    """Example usage of Trie data structure."""
    trie = Trie()
    words = ["apple", "app", "application", "banana", "band", "bat"]

    for word in words:
        trie.insert(word)

    print("Trie Operations:")
    print(f"Search 'apple': {trie.search('apple')}")
    print(f"Search 'app': {trie.search('app')}")
    print(f"Search 'appl': {trie.search('appl')}")
    print(f"Starts with 'app': {trie.starts_with('app')}")
    print(f"Words with prefix 'app': {trie.get_words_with_prefix('app')}")


def lru_cache_example():
    """Example usage of LRU Cache."""
    cache = LRUCache(3)

    print("LRU Cache Operations:")
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")

    print(f"Get 'key1': {cache.get('key1')}")

    # This will evict key2 as key1 was recently accessed
    cache.put("key4", "value4")

    print(f"Get 'key2': {cache.get('key2')}")  # Should return None
    print(f"Get 'key3': {cache.get('key3')}")
    print(f"Get 'key4': {cache.get('key4')}")


def run_all_examples():
    """Run all tree algorithm examples."""
    print("=" * 50)
    print("TREE TRAVERSAL EXAMPLES")
    print("=" * 50)
    traversal_example()

    print("\n" + "=" * 50)
    print("TREE PROPERTIES EXAMPLES")
    print("=" * 50)
    properties_example()

    print("\n" + "=" * 50)
    print("BALANCED TREE EXAMPLES")
    print("=" * 50)
    balanced_tree_example()

    print("\n" + "=" * 50)
    print("SEGMENT TREE EXAMPLES")
    print("=" * 50)
    segment_tree_example()

    print("\n" + "=" * 50)
    print("FENWICK TREE EXAMPLES")
    print("=" * 50)
    fenwick_tree_example()

    print("\n" + "=" * 50)
    print("TRIE EXAMPLES")
    print("=" * 50)
    trie_example()

    print("\n" + "=" * 50)
    print("LRU CACHE EXAMPLES")
    print("=" * 50)
    lru_cache_example()


if __name__ == "__main__":
    run_all_examples()