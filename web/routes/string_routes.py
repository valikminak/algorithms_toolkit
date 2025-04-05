# routes/string_algorithms_routes.py
from flask import Blueprint, jsonify, request

from kosmos.strings.pattern_matching import kmp_search, rabin_karp_search, boyer_moore_search
from kosmos.strings.aho_corasick import aho_corasick_search
from kosmos.strings.suffix_tree import SuffixTree

string_algorithms_routes = Blueprint('string_algorithms_routes', __name__)


@string_algorithms_routes.route('/api/string/algorithms')
def get_string_algorithms():
    """Return available string algorithms"""
    algorithms = [
        {"id": "kmp", "name": "Knuth-Morris-Pratt", "complexity": "O(n+m)"},
        {"id": "rabin_karp", "name": "Rabin-Karp", "complexity": "O(n+m) avg, O(nm) worst"},
        {"id": "boyer_moore", "name": "Boyer-Moore", "complexity": "O(n+m) best, O(nm) worst"},
        {"id": "aho_corasick", "name": "Aho-Corasick", "complexity": "O(n+m+z)"},
        {"id": "suffix_tree", "name": "Suffix Tree", "complexity": "O(n)"}
    ]
    return jsonify(algorithms)


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
            'algorithm': 'rabin_karp',
            'category': 'string',
            'text': text,
            'pattern': pattern,
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
            if j < len(pattern) and pattern[j] == text[i]:
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
            elif i < len(text) and (j >= len(pattern) or pattern[j] != text[i]):
                steps.append({
                    'text_pos': i,
                    'pattern_pos': j if j < len(pattern) else None,
                    'match': False,
                    'characters': f"{text[i]} != {pattern[j] if j < len(pattern) else ''}",
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
            'algorithm': 'kmp',
            'category': 'string',
            'text': text,
            'pattern': pattern,
            'matches': matches,
            'lps': lps,
            'lps_steps': lps_steps,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@string_algorithms_routes.route('/api/string/boyer_moore', methods=['POST'])
def boyer_moore():
    """
    Boyer-Moore string matching algorithm implementation
    Input: JSON with text and pattern
    Output: JSON with matches and steps
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        pattern = data.get('pattern', '')

        matches = boyer_moore_search(text, pattern)

        # Generate bad character table
        def compute_bad_char(pattern):
            m = len(pattern)
            # Initialize bad character table
            bad_char = {}

            # Fill the table with pattern indices
            for i in range(m):
                bad_char[pattern[i]] = i

            return bad_char

        bad_char = compute_bad_char(pattern)

        # Generate visualization steps
        steps = []
        i = 0  # alignment position in text

        while i <= len(text) - len(pattern):
            shift = 0
            mismatched = False

            # Start comparing from the end of pattern
            for j in range(len(pattern) - 1, -1, -1):
                align_pos = i + j

                if align_pos >= len(text):
                    # Out of bounds
                    continue

                if pattern[j] != text[align_pos]:
                    # Character mismatch - compute shift
                    bc_shift = j - bad_char.get(text[align_pos], -1)
                    shift = max(1, bc_shift)
                    mismatched = True

                    steps.append({
                        'position': i,
                        'mismatch_position': j,
                        'text_position': align_pos,
                        'pattern_char': pattern[j],
                        'text_char': text[align_pos],
                        'shift': shift,
                        'matches_so_far': matches[:matches.index(i)] if i in matches else [],
                        'info': f"Mismatch: '{pattern[j]}' vs '{text[align_pos]}', shift by {shift}"
                    })
                    break

            if not mismatched:
                # Found a match
                steps.append({
                    'position': i,
                    'match': True,
                    'matches_so_far': matches[:matches.index(i) + 1] if i in matches else [],
                    'info': f"Match found at position {i}"
                })
                shift = 1  # Shift by 1 to find next match

            i += shift

        return jsonify({
            'algorithm': 'boyer_moore',
            'category': 'string',
            'text': text,
            'pattern': pattern,
            'matches': matches,
            'bad_char_table': bad_char,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@string_algorithms_routes.route('/api/string/aho_corasick', methods=['POST'])
def aho_corasick():
    """
    Aho-Corasick multiple pattern matching algorithm implementation
    Input: JSON with text and patterns
    Output: JSON with matches and steps
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        patterns = data.get('patterns', [])

        if not text or not patterns:
            return jsonify({'error': 'Text and patterns are required'}), 400

        # Use implementation from kosmos library
        matches = aho_corasick_search(text, patterns)

        # Generate visualization steps
        steps = []

        # Simplified animation for Aho-Corasick
        # Step 1: Building the automaton
        steps.append({
            'phase': 'build',
            'info': 'Building the trie with patterns',
            'patterns': patterns,
            'matches_so_far': {}
        })

        # Step 2: Building failure links (simplified)
        steps.append({
            'phase': 'failure',
            'info': 'Computing failure links',
            'patterns': patterns,
            'matches_so_far': {}
        })

        # Step 3: Searching text
        current_matches = {}
        for i, char in enumerate(text):
            step_matches = {}

            # Check which patterns could match at this position
            for pattern in patterns:
                if i >= len(pattern) - 1 and text[i - len(pattern) + 1:i + 1] == pattern:
                    pos = i - len(pattern) + 1
                    if pattern not in step_matches:
                        step_matches[pattern] = []
                    step_matches[pattern].append(pos)

                    if pattern not in current_matches:
                        current_matches[pattern] = []
                    current_matches[pattern].append(pos)

            steps.append({
                'phase': 'search',
                'position': i,
                'character': char,
                'new_matches': step_matches,
                'matches_so_far': {k: v[:] for k, v in current_matches.items()},
                'info': f"Processing character '{char}' at position {i}"
            })

        return jsonify({
            'algorithm': 'aho_corasick',
            'category': 'string',
            'text': text,
            'patterns': patterns,
            'matches': matches,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@string_algorithms_routes.route('/api/string/suffix_tree', methods=['POST'])
def suffix_tree():
    """
    Ukkonen's Suffix Tree algorithm implementation
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

        # Create steps for visualization
        steps = []

        # Add each suffix step by step for visualization
        for i in range(len(text)):
            suffix = text[i:]

            step = {
                'suffix': suffix,
                'position': i,
                'operations': []
            }

            # Track operations for this suffix (simplified version)
            operation = {
                'type': 'new_leaf' if i == 0 else 'traverse',
                'edge': suffix[0] if i == 0 else ''
            }

            step['operations'].append(operation)
            steps.append(step)

        # Create a real suffix tree for the result
        tree = SuffixTree()
        tree.build(text)

        # Extract suffixes from tree
        suffixes = []
        for i in range(len(text)):
            suffixes.append({
                'suffix_index': i,
                'suffix': text[i:]
            })

        return jsonify({
            'algorithm': 'suffix_tree',
            'category': 'string',
            'text': text[:-1],  # Remove the terminator for display
            'tree': {'children': {}},  # Simplified tree structure for visualization
            'steps': steps,
            'suffixes': suffixes
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500