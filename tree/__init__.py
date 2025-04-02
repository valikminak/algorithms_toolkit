# tree/__init__.py

from algorithms_toolkit.tree.base import TreeNode, BinaryTreeNode
from algorithms_toolkit.tree.traversal import (
    binary_tree_inorder_traversal, binary_tree_preorder_traversal,
    binary_tree_postorder_traversal, binary_tree_levelorder_traversal,
    morris_inorder_traversal, morris_preorder_traversal
)
from algorithms_toolkit.tree.properties import (
    tree_height, binary_tree_height, binary_tree_size, binary_tree_is_balanced,
    binary_tree_is_bst, binary_tree_lowest_common_ancestor,
    binary_tree_serialize, binary_tree_deserialize
)
from algorithms_toolkit.tree.balanced import (
    RedBlackTree, RedBlackNode, Color, AVLTree, AVLNode
)
from algorithms_toolkit.tree.segment_tree import (
    SegmentTree, LazySegmentTree, create_sum_segment_tree,
    create_min_segment_tree, create_max_segment_tree,
    create_gcd_segment_tree, create_lazy_sum_segment_tree
)
from algorithms_toolkit.tree.fenwick_tree import (
    FenwickTree, FenwickTree2D, FenwickTreeRangeUpdate, FenwickTreeRangeQuery
)
from algorithms_toolkit.tree.trie import (
    TrieNode, Trie, CompressedTrieNode, CompressedTrie
)
from algorithms_toolkit.tree.lru_cache import (
    LRUCache, LFUCache
)

__all__ = [
    # Base
    'TreeNode', 'BinaryTreeNode',

    # Traversal
    'binary_tree_inorder_traversal', 'binary_tree_preorder_traversal',
    'binary_tree_postorder_traversal', 'binary_tree_levelorder_traversal',
    'morris_inorder_traversal', 'morris_preorder_traversal',

    # Properties
    'tree_height', 'binary_tree_height', 'binary_tree_size', 'binary_tree_is_balanced',
    'binary_tree_is_bst', 'binary_tree_lowest_common_ancestor',
    'binary_tree_serialize', 'binary_tree_deserialize',

    # Balanced Trees
    'RedBlackTree', 'RedBlackNode', 'Color', 'AVLTree', 'AVLNode',

    # Segment Tree
    'SegmentTree', 'LazySegmentTree', 'create_sum_segment_tree',
    'create_min_segment_tree', 'create_max_segment_tree',
    'create_gcd_segment_tree', 'create_lazy_sum_segment_tree',

    # Fenwick Tree
    'FenwickTree', 'FenwickTree2D', 'FenwickTreeRangeUpdate', 'FenwickTreeRangeQuery',

    # Trie
    'TrieNode', 'Trie', 'CompressedTrieNode', 'CompressedTrie',

    # LRU Cache
    'LRUCache', 'LFUCache'
]