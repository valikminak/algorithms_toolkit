# dp/sequence.py
from typing import List, Dict, Tuple, Set
import bisect


def longest_increasing_subsequence(nums: List[int]) -> List[int]:
    """
    Find the longest increasing subsequence in a list of numbers.

    Args:
        nums: List of numbers

    Returns:
        The longest increasing subsequence
    """
    if not nums:
        return []

    n = len(nums)

    # dp[i] = length of LIS ending at index i
    dp = [1] * n

    # prev[i] = previous index in the LIS ending at index i
    prev = [-1] * n

    # Fill DP table
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
                prev[i] = j

    # Find the index with maximum LIS length
    max_length = max(dp)
    max_index = dp.index(max_length)

    # Reconstruct the LIS
    lis = []
    while max_index != -1:
        lis.append(nums[max_index])
        max_index = prev[max_index]

    return list(reversed(lis))


def longest_common_subsequence(s1: str, s2: str) -> str:
    """
    Find the longest common subsequence of two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The longest common subsequence
    """
    m, n = len(s1), len(s2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS
    i, j = m, n
    lcs = []

    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(lcs))


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Find the longest common substring of two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The longest common substring
    """
    m, n = len(s1), len(s2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Variables to keep track of the maximum length and ending position
    max_length = 0
    end_pos = 0

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0

    # Extract the longest common substring
    start = end_pos - max_length
    return s1[start:end_pos]


def longest_palindromic_subsequence(s: str) -> int:
    """
    Find the length of the longest palindromic subsequence in a string.

    Args:
        s: Input string

    Returns:
        Length of the longest palindromic subsequence
    """
    n = len(s)

    # Create DP table
    # dp[i][j] = length of LPS in s[i:j+1]
    dp = [[0] * n for _ in range(n)]

    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = 1

    # Fill the DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    return dp[0][n - 1]


def count_different_palindromic_subsequences(s: str) -> int:
    """
    Count different palindromic subsequences in a string.

    Args:
        s: Input string

    Returns:
        Number of different palindromic subsequences
    """
    n = len(s)
    MOD = 10 ** 9 + 7

    # Create DP table
    # dp[i][j] = number of different palindromic subsequences in s[i:j+1]
    dp = [[0] * n for _ in range(n)]

    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = 1

    # Fill the DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] != s[j]:
                dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1]) % MOD
            else:
                # Find the positions of the next and previous occurrence of s[i]
                left = i + 1
                right = j - 1

                while left <= right and s[left] != s[i]:
                    left += 1

                while left <= right and s[right] != s[i]:
                    right -= 1

                if left > right:
                    # No occurrence of s[i] in between
                    dp[i][j] = (2 * dp[i + 1][j - 1] + 2) % MOD
                elif left == right:
                    # One occurrence of s[i] in between
                    dp[i][j] = (2 * dp[i + 1][j - 1] + 1) % MOD
                else:
                    # More than one occurrence of s[i] in between
                    dp[i][j] = (2 * dp[i + 1][j - 1] - dp[left + 1][right - 1]) % MOD

    return (dp[0][n - 1] + MOD) % MOD  # Ensure positive result


def word_break(s: str, wordDict: List[str]) -> bool:
    """
    Determine if a string can be segmented into words from a dictionary.

    Args:
        s: Input string
        wordDict: List of dictionary words

    Returns:
        True if the string can be segmented, False otherwise
    """
    n = len(s)
    word_set = set(wordDict)

    # dp[i] = True if s[0:i] can be segmented
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string can be segmented

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]


def partition_equal_subset_sum(nums: List[int]) -> bool:
    """
    Determine if a list can be partitioned into two subsets with equal sum.

    Args:
        nums: List of integers

    Returns:
        True if the list can be partitioned, False otherwise
    """
    total_sum = sum(nums)

    # If the sum is odd, it cannot be partitioned
    if total_sum % 2 != 0:
        return False

    target = total_sum // 2
    n = len(nums)

    # dp[i][j] = True if a subset of nums[0..i-1] has sum j
    dp = [[False] * (target + 1) for _ in range(n + 1)]

    # Empty subset has sum 0
    for i in range(n + 1):
        dp[i][0] = True

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            if j < nums[i - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]

    return dp[n][target]


def distinct_subsequences(s: str, t: str) -> int:
    """
    Count the number of distinct subsequences of s that equal t.

    Args:
        s: Source string
        t: Target string

    Returns:
        Number of distinct subsequences
    """
    m, n = len(s), len(t)

    # dp[i][j] = number of distinct subsequences of s[0...i-1] that equal t[0...j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Empty string t is a subsequence of any string s once
    for i in range(m + 1):
        dp[i][0] = 1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If current characters match, we have two options:
            # 1. Use the character in s to match t
            # 2. Don't use the character in s
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
            else:
                # Only option is to not use the character in s
                dp[i][j] = dp[i - 1][j]

    return dp[m][n]


def longest_repeating_substring(s: str) -> str:
    """
    Find the longest repeating substring in a string.

    Args:
        s: Input string

    Returns:
        The longest repeating substring
    """
    n = len(s)

    # dp[i][j] = length of longest common substring ending at s[i-1] and s[j-1]
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    # Variables to keep track of the maximum length and ending position
    max_length = 0
    end_pos = 0

    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if s[i - 1] == s[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                if dp[i][j] > max_length:
                    # Check if it's a valid repeating substring (not overlapping)
                    if i - dp[i][j] != j - dp[i][j]:
                        max_length = dp[i][j]
                        end_pos = i

    if max_length == 0:
        return ""

    return s[end_pos - max_length:end_pos]


def longest_palindrome(s: str) -> str:
    """
    Find the longest palindromic substring in a string.

    Args:
        s: Input string

    Returns:
        The longest palindromic substring
    """
    if not s:
        return ""

    n = len(s)
    start, max_len = 0, 1

    # dp[i][j] = True if s[i:j+1] is a palindrome
    dp = [[False] * n for _ in range(n)]

    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = True

    # Check for substrings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2

    # Check for substrings of length 3 or more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            # Check if the substring s[i:j+1] is a palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length

    return s[start:start + max_len]