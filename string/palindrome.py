# string/palindrome.py
from typing import List, Tuple


def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome (reads the same forward and backward).

    Args:
        s: The string to check

    Returns:
        True if the string is a palindrome, False otherwise
    """
    # Convert to lowercase and remove non-alphanumeric characters
    s = ''.join(c.lower() for c in s if c.isalnum())

    # Check if the string is equal to its reverse
    return s == s[::-1]


def longest_palindromic_substring(s: str) -> str:
    """
    Find the longest palindromic substring in a string.

    Args:
        s: The input string

    Returns:
        The longest palindromic substring
    """
    if not s:
        return ""

    start = 0
    max_length = 1

    # Helper function to expand around center
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    for i in range(len(s)):
        # Expand around center (odd length)
        length1 = expand_around_center(i, i)

        # Expand around center (even length)
        length2 = expand_around_center(i, i + 1)

        # Get maximum length
        length = max(length1, length2)

        # Update result if needed
        if length > max_length:
            max_length = length
            start = i - (length - 1) // 2

    return s[start:start + max_length]


def manacher_algorithm(s: str) -> str:
    """
    Manacher's algorithm for finding the longest palindromic substring in linear time.

    Args:
        s: The input string

    Returns:
        The longest palindromic substring
    """
    if not s:
        return ""

    # Preprocess the string
    # Insert special character between each character and at boundaries
    # This handles both odd and even length palindromes
    t = '#' + '#'.join(s) + '#'
    n = len(t)

    # p[i] = radius of palindrome centered at i
    p = [0] * n

    center = 0  # Center of the rightmost palindrome
    right = 0  # Right boundary of the rightmost palindrome

    for i in range(n):
        # Initial value for p[i] using symmetry
        if right > i:
            p[i] = min(right - i, p[2 * center - i])

        # Expand palindrome centered at i
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1

        # Update center and right boundary if needed
        if i + p[i] > right:
            center = i
            right = i + p[i]

    # Find the maximum palindrome length
    max_len = max(p)
    center_index = p.index(max_len)

    # Convert back to original string indices
    start = (center_index - max_len) // 2
    end = start + max_len

    return s[start:end]


def count_palindromic_substrings(s: str) -> int:
    """
    Count the number of palindromic substrings in a string.

    Args:
        s: The input string

    Returns:
        The count of palindromic substrings
    """
    if not s:
        return 0

    count = 0
    n = len(s)

    # Helper function to count palindromes expanding from center
    def count_palindromes(left, right):
        count = 0
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count

    # Check each possible center
    for i in range(n):
        # Odd length palindromes with center at i
        count += count_palindromes(i, i)

        # Even length palindromes with center between i and i+1
        count += count_palindromes(i, i + 1)

    return count


def longest_palindromic_subsequence(s: str) -> str:
    """
    Find the longest palindromic subsequence in a string.

    A subsequence is a sequence that can be derived from another sequence
    by deleting some elements without changing the order of the remaining elements.

    Args:
        s: The input string

    Returns:
        The longest palindromic subsequence
    """
    n = len(s)
    if n <= 1:
        return s

    # dp[i][j] = length of LPS from s[i] to s[j]
    dp = [[0] * n for _ in range(n)]

    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = 1

    # Fill the dp table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    # Reconstruct the subsequence
    result = []
    i, j = 0, n - 1

    while i <= j:
        if s[i] == s[j]:
            result.append(s[i])
            i += 1
            j -= 1
        elif dp[i + 1][j] >= dp[i][j - 1]:
            i += 1
        else:
            j -= 1

    # For odd length palindromes, the middle character is counted twice
    subsequence = ''.join(result)
    if dp[0][n - 1] % 2 == 1:
        subsequence = subsequence[:-1]

    return subsequence + ''.join(reversed(result))


def is_palindrome_with_one_edit(s: str) -> bool:
    """
    Check if a string can become a palindrome by removing at most one character.

    Args:
        s: The input string

    Returns:
        True if the string can become a palindrome with at most one deletion, False otherwise
    """

    # Check if characters match from both ends
    def is_palindrome_range(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True

    left, right = 0, len(s) - 1

    while left < right:
        if s[left] != s[right]:
            # Try removing character at left or right and check if the rest is a palindrome
            return is_palindrome_range(left + 1, right) or is_palindrome_range(left, right - 1)
        left += 1
        right -= 1

    return True  # Already a palindrome


def shortest_palindrome(s: str) -> str:
    """
    Find the shortest palindrome by adding characters to the beginning of the string.

    Args:
        s: The input string

    Returns:
        The shortest palindrome by adding characters to the beginning
    """
    if not s or is_palindrome(s):
        return s

    # Find the longest palindrome prefix
    n = len(s)
    rev = s[::-1]

    # Concatenate s with a special character and its reverse
    # to find the longest palindrome prefix using KMP
    new_s = s + '#' + rev
    m = len(new_s)

    # Compute LPS array
    lps = [0] * m
    i, j = 1, 0

    while i < m:
        if new_s[i] == new_s[j]:
            j += 1
            lps[i] = j
            i += 1
        elif j > 0:
            j = lps[j - 1]
        else:
            lps[i] = 0
            i += 1

    # Length of the longest palindrome prefix
    longest_prefix_length = lps[-1]

    # Add the reversed suffix to the beginning
    return rev[:n - longest_prefix_length] + s