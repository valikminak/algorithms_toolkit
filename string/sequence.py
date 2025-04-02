from typing import List, Dict, Tuple, Set
import bisect


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


def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate the edit distance (Levenshtein distance) between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        The minimum number of operations (insert, delete, replace) to transform s1 into s2
    """
    m, n = len(s1), len(s2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Delete
                    dp[i][j - 1],  # Insert
                    dp[i - 1][j - 1]  # Replace
                )

    return dp[m][n]


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


def longest_increasing_subsequence_optimized(nums: List[int]) -> List[int]:
    """
    Find the longest increasing subsequence in a list of numbers using binary search.

    Args:
        nums: List of numbers

    Returns:
        The longest increasing subsequence
    """
    if not nums:
        return []

    n = len(nums)

    # tails[i] = smallest value that can end an increasing subsequence of length i+1
    tails = []

    # prev[i] = previous index in the LIS ending at index i
    prev = [-1] * n

    # Indices mapping tails positions to original indices
    indices = []

    for i, num in enumerate(nums):
        # Binary search to find the position to insert nums[i]
        pos = bisect.bisect_left(tails, num)

        if pos == len(tails):
            # Append to tails if num is greater than all elements
            tails.append(num)
            indices.append(i)
        else:
            # Replace the element at pos
            tails[pos] = num
            indices[pos] = i

        # Update prev array
        if pos > 0:
            prev[i] = indices[pos - 1]

    # Reconstruct the LIS
    lis = []
    curr = indices[-1]

    while curr != -1:
        lis.append(nums[curr])
        curr = prev[curr]

    return list(reversed(lis))


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


def minimum_window_substring(s: str, t: str) -> str:
    """
    Find the minimum window in s which contains all the characters in t.

    Args:
        s: The source string
        t: The target string

    Returns:
        The minimum window substring containing all characters in t, or empty string if not found
    """
    if not s or not t or len(s) < len(t):
        return ""

    # Count characters in t
    t_count = {}
    for char in t:
        t_count[char] = t_count.get(char, 0) + 1

    # Variables to track the minimum window
    required = len(t_count)
    formed = 0
    window_counts = {}

    # Variables for the result
    min_len = float('inf')
    result = ""

    # Use two pointers: left and right
    left, right = 0, 0

    while right < len(s):
        # Add the current character to the window
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        # Check if we've met the count for this character
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1

        # Try to minimize the window by moving left pointer
        while left <= right and formed == required:
            char = s[left]

            # Update the result if this is a smaller window
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = s[left:right + 1]

            # Remove the leftmost character from the window
            window_counts[char] -= 1

            # Check if this character is part of t and removal affects the formed count
            if char in t_count and window_counts[char] < t_count[char]:
                formed -= 1

            left += 1

        right += 1

    return result if min_len != float('inf') else ""


def longest_repeating_character_replacement(s: str, k: int) -> int:
    """
    Find the length of the longest substring containing the same letter after replacing at most k characters.

    Args:
        s: The input string
        k: Maximum number of characters to replace

    Returns:
        Length of the longest substring after replacement
    """
    if not s:
        return 0

    n = len(s)
    max_length = 0
    max_count = 0
    char_counts = {}

    left = 0
    for right in range(n):
        # Add the current character to the window
        char = s[right]
        char_counts[char] = char_counts.get(char, 0) + 1

        # Track the most frequent character in the current window
        max_count = max(max_count, char_counts[char])

        # If the number of replacements needed exceeds k, shrink the window
        if (right - left + 1) - max_count > k:
            char_counts[s[left]] -= 1
            left += 1

        # Update the max length
        max_length = max(max_length, right - left + 1)

    return max_length


def longest_substring_without_repeating_characters(s: str) -> str:
    """
    Find the longest substring without repeating characters.

    Args:
        s: The input string

    Returns:
        The longest substring without repeating characters
    """
    if not s:
        return ""

    n = len(s)
    char_index = {}  # Track the last position of each character

    start = 0
    max_length = 0
    max_start = 0

    for i, char in enumerate(s):
        # If the character is already in the current substring, update the start position
        if char in char_index and start <= char_index[char]:
            start = char_index[char] + 1
        # Update the max length if the current substring is longer
        elif i - start + 1 > max_length:
            max_length = i - start + 1
            max_start = start

        # Update the character's last position
        char_index[char] = i

    return s[max_start:max_start + max_length]


def is_subsequence(s: str, t: str) -> bool:
    """
    Check if s is a subsequence of t.

    Args:
        s: First string
        t: Second string

    Returns:
        True if s is a subsequence of t, False otherwise
    """
    if not s:
        return True

    if not t:
        return False

    i, j = 0, 0

    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1

    return i == len(s)