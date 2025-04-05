# routes/string_algorithms_routes.py
from flask import Blueprint, jsonify, request

from kosmos.strings.pattern_matching import kmp_search, rabin_karp_search

# Note: We would import suffix tree functionality here if it exists

string_algorithms_routes = Blueprint('string_algorithms_routes', __name__)


@string_algorithms_routes.route('/api/string/rabin_karp', methods=['POST'])
def rabin_karp():
    """
    Rabin-Karp string matching algorithm implementation
    Input: JSON with text and pattern
    Output: JSON with matches and steps
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        pattern = data.get('pattern', '')

        # Using the existing implementation
        matches = rabin_karp_search(text, pattern)

        # Generate steps for visualization
        steps = []
        prime = 101  # Same as used in the algorithm

        def calculate_hash(string, length):
            """Calculate the hash value for a string."""
            result = 0
            for i in range(length):
                result += ord(string[i]) * (prime ** (length - i - 1))
            return result % prime

        pattern_length = len(pattern)

        # Calculate pattern hash
        pattern_hash = calculate_hash(pattern, pattern_length)

        # Calculate initial text window hash
        text_hash = calculate_hash(text[:pattern_length], pattern_length) if len(text) >= pattern_length else 0

        # Initial step
        steps.append({
            'position': 0,
            'text_window': text[0:pattern_length] if len(text) >= pattern_length else "",
            'pattern_hash': pattern_hash,
            'text_hash': text_hash,
            'hash_match': text_hash == pattern_hash,
            'string_match': text[0:pattern_length] == pattern if len(text) >= pattern_length else False,
            'matches_so_far': []
        })

        matches_so_far = []
        if len(text) >= pattern_length and text_hash == pattern_hash and text[0:pattern_length] == pattern:
            matches_so_far.append(0)

        # Continue with window slides
        for i in range(1, len(text) - pattern_length + 1):
            # Update hash by removing leading digit and adding trailing digit
            text_hash = ((text_hash - ord(text[i - 1]) * (prime ** (pattern_length - 1))) * prime +
                         ord(text[i + pattern_length - 1])) % prime

            steps.append({
                'position': i,
                'text_window': text[i:i + pattern_length],
                'pattern_hash': pattern_hash,
                'text_hash': text_hash,
                'hash_match': text_hash == pattern_hash,
                'string_match': text[i:i + pattern_length] == pattern,
                'matches_so_far': matches_so_far.copy()
            })

            if text_hash == pattern_hash and text[i:i + pattern_length] == pattern:
                matches_so_far.append(i)

        return jsonify({
            'matches': matches,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@string_algorithms_routes.route('/api/string/kmp', methods=['POST'])
def knuth_morris_pratt():
    """
    Knuth-Morris-Pratt string matching algorithm implementation
    Input: JSON with text and pattern
    Output: JSON with matches and steps
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        pattern = data.get('pattern', '')

        # Using the existing implementation
        matches = kmp_search(text, pattern)

        # Generate LPS array with steps
        def compute_lps(pattern):
            m = len(pattern)
            lps = [0] * m

            length = 0
            i = 1
            lps_steps = []

            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    lps_steps.append({
                        'i': i,
                        'length': length,
                        'lps': lps.copy(),
                        'match': True,
                        'characters': f"{pattern[i]} = {pattern[length - 1]}"
                    })
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                        lps_steps.append({
                            'i': i,
                            'length': length,
                            'lps': lps.copy(),
                            'match': False,
                            'fallback': True,
                            'message': f"Mismatch, fallback to lps[{length - 1}]"
                        })
                    else:
                        lps[i] = 0
                        lps_steps.append({
                            'i': i,
                            'length': length,
                            'lps': lps.copy(),
                            'match': False,
                            'fallback': False,
                            'message': "No match, no fallback"
                        })
                        i += 1

            return lps, lps_steps

        lps, lps_steps = compute_lps(pattern)

        # Generate search steps
        steps = []
        i = 0  # Index for text
        j = 0  # Index for pattern
        matches_so_far = []

        while i < len(text):
            # Match characters
            if pattern[j] == text[i]:
                i += 1
                j += 1
                steps.append({
                    'text_pos': i - 1,
                    'pattern_pos': j - 1,
                    'match': True,
                    'j': j,
                    'text_window': text[i - j:i],
                    'matches_so_far': matches_so_far.copy()
                })

            # Found a match
            if j == len(pattern):
                matches_so_far.append(i - j)
                j = lps[j - 1]
                steps.append({
                    'text_pos': i,
                    'pattern_pos': j,
                    'complete_match': True,
                    'match_position': i - len(pattern),
                    'j': j,
                    'fallback': True,
                    'matches_so_far': matches_so_far.copy()
                })

            # Mismatch after some matches
            elif i < len(text) and pattern[j] != text[i]:
                steps.append({
                    'text_pos': i,
                    'pattern_pos': j,
                    'match': False,
                    'characters': f"{text[i]} != {pattern[j]}",
                    'matches_so_far': matches_so_far.copy()
                })

                if j != 0:
                    j = lps[j - 1]
                    steps.append({
                        'text_pos': i,
                        'pattern_pos': j,
                        'fallback': True,
                        'j': j,
                        'matches_so_far': matches_so_far.copy()
                    })
                else:
                    i += 1

        return jsonify({
            'matches': matches,
            'steps': steps,
            'lps': lps,
            'lps_steps': lps_steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@string_algorithms_routes.route('/api/string/suffix_tree', methods=['POST'])
def suffix_tree():
    """
    Ukkonen's Suffix Tree algorithm implementation (simplified)
    Input: JSON with text
    Output: JSON with tree representation and steps
    """
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'Text is required'}), 400

        # Add a unique terminator to the text
        text = text + '$'

        # Simple SuffixTrie implementation for visualization
        class Node:
            def __init__(self):
                self.children = {}
                self.suffix_link = None
                self.start = -1
                self.end = -1
                self.suffix_index = -1

            def to_dict(self):
                return {
                    'children': {k: v.to_dict() for k, v in self.children.items()},
                    'start': self.start,
                    'end': self.end,
                    'suffix_index': self.suffix_index,
                    'has_suffix_link': self.suffix_link is not None
                }

        # Initialize the tree
        root = Node()

        steps = []

        # Simple Suffix Tree construction using naive approach for visualization
        for i in range(len(text)):
            suffix = text[i:]
            current = root
            j = 0

            step = {
                'suffix': suffix,
                'position': i,
                'operations': []
            }

            while j < len(suffix):
                char = suffix[j]

                if char in current.children:
                    child = current.children[char]
                    edge_length = child.end - child.start + 1

                    # Check for match along the edge
                    k = 0
                    while k < edge_length and j + k < len(suffix) and suffix[j + k] == text[child.start + k]:
                        k += 1

                    if k == edge_length:
                        # Fully matched the edge, move to next node
                        current = child
                        j += edge_length
                        step['operations'].append({
                            'type': 'traverse',
                            'edge': text[child.start:child.end + 1],
                            'full_match': True
                        })
                    else:
                        # Split the edge
                        split = Node()
                        split.start = child.start
                        split.end = child.start + k - 1

                        # Adjust old child
                        child.start = child.start + k
                        split.children[text[child.start]] = child

                        # Create new leaf
                        leaf = Node()
                        leaf.start = i + j + k
                        leaf.end = len(text) - 1
                        leaf.suffix_index = i
                        split.children[suffix[j + k]] = leaf

                        # Update parent's reference
                        current.children[char] = split

                        step['operations'].append({
                            'type': 'split',
                            'edge': text[split.start:split.end + 1],
                            'match_length': k,
                            'new_leaf': text[leaf.start:leaf.end + 1]
                        })
                        break
                else:
                    # Create a new leaf node
                    leaf = Node()
                    leaf.start = i + j
                    leaf.end = len(text) - 1
                    leaf.suffix_index = i
                    current.children[char] = leaf

                    step['operations'].append({
                        'type': 'new_leaf',
                        'edge': text[leaf.start:leaf.end + 1]
                    })
                    break

            steps.append(step)

        # Traverse the tree to collect all suffixes
        suffixes = []

        def traverse(node, path=''):
            if node.suffix_index != -1:
                suffixes.append({
                    'suffix_index': node.suffix_index,
                    'suffix': text[node.suffix_index:]
                })

            for char, child in sorted(node.children.items()):
                edge = text[child.start:child.end + 1]
                traverse(child, path + edge)

        traverse(root)

        return jsonify({
            'tree': root.to_dict(),
            'text': text,
            'steps': steps,
            'suffixes': suffixes
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500