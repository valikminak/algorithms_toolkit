from typing import List, Tuple, Dict
import collections


def burrows_wheeler_transform(s: str) -> Tuple[str, int]:
    """
    Perform the Burrows-Wheeler Transform (BWT) on a string.

    The BWT is a reversible transformation that rearranges characters
    to improve compression.

    Args:
        s: Input string

    Returns:
        Tuple of (transformed_string, original_index) where:
        - transformed_string: The BWT of the input string
        - original_index: The index of the original string in the sorted rotations
    """
    # Append a sentinel character
    s = s + "$"

    # Generate all rotations of s
    rotations = []
    for i in range(len(s)):
        rotations.append(s[i:] + s[:i])

    # Sort the rotations
    sorted_rotations = sorted(rotations)

    # Find the index of the original string
    original_index = sorted_rotations.index(s)

    # Get the last character of each rotation
    bwt = ''.join(r[-1] for r in sorted_rotations)

    return bwt, original_index


def inverse_burrows_wheeler_transform(bwt: str, original_index: int) -> str:
    """
    Perform the inverse Burrows-Wheeler Transform to recover the original string.

    Args:
        bwt: The BWT of the original string
        original_index: The index of the original string in the sorted rotations

    Returns:
        The original string
    """
    n = len(bwt)

    # Create sorted string of first characters
    first_column = ''.join(sorted(bwt))

    # Create mapping from character and occurrence to position in first column
    char_counts = {}
    first_occ = {}

    for i, c in enumerate(first_column):
        if c not in char_counts:
            char_counts[c] = 0
            first_occ[c] = i
        else:
            char_counts[c] += 1

    # Create mapping for each character in bwt to its position in first column
    next_index = []
    char_counts = {}

    for c in bwt:
        if c not in char_counts:
            char_counts[c] = 0
            next_index.append(first_occ[c])
        else:
            next_index.append(first_occ[c] + char_counts[c])
            char_counts[c] += 1

    # Reconstruct the original string
    result = []
    i = original_index

    for _ in range(n - 1):  # Skip the sentinel character
        i = next_index[i]
        result.append(first_column[i])

    # Remove the sentinel character
    return ''.join(result)


def move_to_front_encode(s: str) -> List[int]:
    """
    Perform Move-to-Front encoding on a string.

    This is often used as a stage in BWT-based compression.

    Args:
        s: Input string

    Returns:
        List of integers representing the encoded string
    """
    # Initialize the alphabet
    alphabet = list(set(s))
    alphabet.sort()

    # Encode the string
    result = []

    for c in s:
        # Find the position of c in the alphabet
        i = alphabet.index(c)
        result.append(i)

        # Move c to the front
        alphabet.pop(i)
        alphabet.insert(0, c)

    return result


def move_to_front_decode(encoded: List[int], alphabet: List[str]) -> str:
    """
    Perform Move-to-Front decoding.

    Args:
        encoded: List of integers representing the encoded string
        alphabet: The initial alphabet

    Returns:
        The decoded string
    """
    result = []

    for i in encoded:
        # Get the character at position i
        c = alphabet[i]
        result.append(c)

        # Move c to the front
        alphabet.pop(i)
        alphabet.insert(0, c)

    return ''.join(result)


def run_length_encode(s: str) -> str:
    """
    Perform Run-Length Encoding (RLE) on a string.

    RLE replaces consecutive repeated characters with the character and the count.

    Args:
        s: Input string

    Returns:
        The encoded string
    """
    if not s:
        return ""

    result = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(s[i - 1] + str(count))
            count = 1

    # Add the last run
    result.append(s[-1] + str(count))

    return ''.join(result)


def run_length_decode(s: str) -> str:
    """
    Perform Run-Length Decoding on a string.

    Args:
        s: Input string

    Returns:
        The decoded string
    """
    result = []
    i = 0

    while i < len(s):
        # Get the character and count
        char = s[i]
        i += 1

        # Extract the count digits
        count_str = ""
        while i < len(s) and s[i].isdigit():
            count_str += s[i]
            i += 1

        count = int(count_str)

        # Repeat the character 'count' times
        result.append(char * count)

    return ''.join(result)


def huffman_coding(s: str) -> Tuple[Dict[str, str], str]:
    """
    Perform Huffman coding on a string.

    Huffman coding assigns variable-length codes to characters based on frequency.

    Args:
        s: Input string

    Returns:
        Tuple of (code_dict, encoded_string) where:
        - code_dict: Dictionary mapping characters to their Huffman codes
        - encoded_string: The Huffman-encoded string
    """
    # Count character frequencies
    freq = collections.Counter(s)

    # If only one character, assign it code '0'
    if len(freq) == 1:
        char = list(freq.keys())[0]
        return {char: '0'}, '0' * len(s)

    # Create a priority queue (heap) of (frequency, node) pairs
    heap = [[f, [c, ""]] for c, f in freq.items()]
    heap.sort(key=lambda x: x[0])

    # Build the Huffman tree
    while len(heap) > 1:
        lo = heap.pop(0)
        hi = heap.pop(0)

        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]

        heap.append([lo[0] + hi[0]] + lo[1:] + hi[1:])
        heap.sort(key=lambda x: x[0])

    # Extract the codes
    code_dict = {char: code for char, code in heap[0][1:]}

    # Encode the string
    encoded_string = ''.join(code_dict[c] for c in s)

    return code_dict, encoded_string


def huffman_decode(encoded: str, code_dict: Dict[str, str]) -> str:
    """
    Decode a Huffman-encoded string.

    Args:
        encoded: The Huffman-encoded string
        code_dict: Dictionary mapping characters to their Huffman codes

    Returns:
        The decoded string
    """
    # Invert the code dictionary
    reverse_dict = {code: char for char, code in code_dict.items()}

    # Decode the string
    result = []
    current_code = ""

    for bit in encoded:
        current_code += bit

        if current_code in reverse_dict:
            result.append(reverse_dict[current_code])
            current_code = ""

    return ''.join(result)


def lzw_compress(s: str) -> List[int]:
    """
    Perform Lempel-Ziv-Welch (LZW) compression on a string.

    Args:
        s: Input string

    Returns:
        List of integers representing the compressed string
    """
    # Initialize the dictionary with single characters
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256

    # Compress the string
    result = []
    current_str = ""

    for c in s:
        new_str = current_str + c

        if new_str in dictionary:
            current_str = new_str
        else:
            # Output the code for current_str
            result.append(dictionary[current_str])

            # Add new_str to the dictionary
            dictionary[new_str] = next_code
            next_code += 1

            # Start a new string
            current_str = c

    # Output the code for the last string
    if current_str:
        result.append(dictionary[current_str])

    return result


def lzw_decompress(compressed: List[int]) -> str:
    """
    Perform Lempel-Ziv-Welch (LZW) decompression.

    Args:
        compressed: List of integers representing the compressed string

    Returns:
        The decompressed string
    """
    # Initialize the dictionary with single characters
    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256

    # Handle empty input
    if not compressed:
        return ""

    # Start with the first code
    result = [dictionary[compressed[0]]]
    current_str = result[0]

    for code in compressed[1:]:
        # Check if the code is in the dictionary
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            # Special case for the pattern like "abababa"
            entry = current_str + current_str[0]
        else:
            raise ValueError(f"Invalid compressed code: {code}")

        # Add the entry to the result
        result.append(entry)

        # Add a new entry to the dictionary
        dictionary[next_code] = current_str + entry[0]
        next_code += 1

        # Update current_str
        current_str = entry

    return ''.join(result)