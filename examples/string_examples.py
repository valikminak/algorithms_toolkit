from string.pattern_matching import kmp_search, rabin_karp_search, z_algorithm
from string.palindrome import is_palindrome, longest_palindromic_substring, manacher_algorithm
from string.sequence import longest_common_subsequence, edit_distance, longest_increasing_subsequence
from string.aho_corasick import AhoCorasick
from string.suffix import SuffixArray
from string.compression import run_length_encode, run_length_decode, huffman_coding, huffman_decode


def pattern_matching_example():
    """Example usage of pattern matching algorithms."""
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"

    print("String Pattern Matching:")
    print(f"Text: {text}")
    print(f"Pattern: {pattern}")

    # KMP Algorithm
    matches = kmp_search(text, pattern)
    print(f"KMP search: Found at indices {matches}")

    # Rabin-Karp Algorithm
    matches = rabin_karp_search(text, pattern)
    print(f"Rabin-Karp search: Found at indices {matches}")

    # Z Algorithm
    # For Z algorithm, we typically concatenate pattern + special_char + text
    combined = pattern + "#" + text
    z_values = z_algorithm(combined)
    matches = [i - len(pattern) - 1 for i in range(len(pattern) + 1, len(combined))
               if z_values[i] == len(pattern)]
    print(f"Z algorithm search: Found at indices {matches}")


def palindrome_example():
    """Example usage of palindrome algorithms."""
    s1 = "racecar"
    s2 = "A man, a plan, a canal: Panama"
    s3 = "babad"

    print("Palindrome Checks:")
    print(f"Is '{s1}' a palindrome? {is_palindrome(s1)}")
    print(f"Is '{s2}' a palindrome? {is_palindrome(s2)}")

    print("\nLongest Palindromic Substring:")
    print(f"For '{s3}': {longest_palindromic_substring(s3)}")
    print(f"Using Manacher's algorithm: {manacher_algorithm(s3)}")


def sequence_example():
    """Example usage of sequence algorithms."""
    s1 = "ABCBDAB"
    s2 = "BDCABA"

    print("Longest Common Subsequence:")
    lcs = longest_common_subsequence(s1, s2)
    print(f"LCS of '{s1}' and '{s2}': '{lcs}'")

    print("\nEdit Distance:")
    distance = edit_distance("kitten", "sitting")
    print(f"Edit distance between 'kitten' and 'sitting': {distance}")

    print("\nLongest Increasing Subsequence:")
    nums = [10, 22, 9, 33, 21, 50, 41, 60, 80]
    lis = longest_increasing_subsequence(nums)
    print(f"LIS of {nums}: {lis}")


def aho_corasick_example():
    """Example usage of Aho-Corasick algorithm."""
    ac = AhoCorasick()
    patterns = ["he", "she", "his", "hers"]

    for pattern in patterns:
        ac.add_pattern(pattern)
    ac.build()

    text = "ahishers"
    results = ac.search(text)

    print("Aho-Corasick Pattern Matching:")
    print(f"Text: {text}")
    print(f"Patterns: {patterns}")
    print(f"Found: {results}")


def suffix_array_example():
    """Example usage of Suffix Array."""
    text = "banana"
    suffix_array = SuffixArray(text)

    print("Suffix Array:")
    print(f"Text: {text}")
    print(f"Suffix Array: {suffix_array.suffix_array}")

    # Search for a pattern
    pattern = "ana"
    matches = suffix_array.search(pattern)
    print(f"Occurrences of '{pattern}': {matches}")

    # Find longest repeated substring
    lrs, positions = suffix_array.longest_repeated_substring()
    print(f"Longest repeated substring: '{lrs}' at positions {positions}")


def compression_example():
    """Example usage of string compression algorithms."""
    text = "AAABBBCCDAA"

    # Run-length encoding
    encoded = run_length_encode(text)
    decoded = run_length_decode(encoded)

    print("Run-Length Encoding:")
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Huffman coding
    code_dict, encoded = huffman_coding(text)
    decoded = huffman_decode(encoded, code_dict)

    print("\nHuffman Coding:")
    print(f"Original: {text} ({len(text) * 8} bits assuming 8 bits per char)")
    print(f"Code dictionary: {code_dict}")
    print(f"Encoded: {encoded} ({len(encoded)} bits)")
    print(f"Decoded: {decoded}")


def run_all_examples():
    """Run all string algorithm examples."""
    print("=" * 50)
    print("PATTERN MATCHING EXAMPLES")
    print("=" * 50)
    pattern_matching_example()

    print("\n" + "=" * 50)
    print("PALINDROME EXAMPLES")
    print("=" * 50)
    palindrome_example()

    print("\n" + "=" * 50)
    print("SEQUENCE EXAMPLES")
    print("=" * 50)
    sequence_example()

    print("\n" + "=" * 50)
    print("AHO-CORASICK EXAMPLES")
    print("=" * 50)
    aho_corasick_example()

    print("\n" + "=" * 50)
    print("SUFFIX ARRAY EXAMPLES")
    print("=" * 50)
    suffix_array_example()

    print("\n" + "=" * 50)
    print("COMPRESSION EXAMPLES")
    print("=" * 50)
    compression_example()


if __name__ == "__main__":
    run_all_examples()