from typing import List


def kmp_search(text: str, pattern: str) -> List[int]:
    """
    Knuth-Morris-Pratt (KMP) algorithm for pattern matching in strings.

    Args:
        text: The text to search in
        pattern: The pattern to search for

    Returns:
        List of starting indices where the pattern is found in the text
    """
    if not pattern:
        return []

    # Compute the LPS (Longest Proper Prefix which is also Suffix) array
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1

        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        return lps

    lps = compute_lps(pattern)

    # Search for the pattern
    results = []
    i = 0  # Index for text
    j = 0  # Index for pattern

    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == len(pattern):
            # Pattern found at index i - j
            results.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results


def rabin_karp_search(text: str, pattern: str, q: int = 101) -> List[int]:
    """
    Rabin-Karp algorithm for pattern matching in strings.

    Args:
        text: The text to search in
        pattern: The pattern to search for
        q: A prime number used for hashing

    Returns:
        List of starting indices where the pattern is found in the text
    """
    if not pattern:
        return []

    n = len(text)
    m = len(pattern)
    results = []

    if m > n:
        return results

    # Hash function: h(s) = (s[0] * d^(m-1) + s[1] * d^(m-2) + ... + s[m-1]) % q
    # where d is the number of characters in the alphabet
    d = 256  # Assuming ASCII

    # Calculate (d^(m-1)) % q
    h = 1
    for _ in range(m - 1):
        h = (h * d) % q

    # Calculate hash values for pattern and first window of text
    p = 0  # Hash value for pattern
    t = 0  # Hash value for current window of text

    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    # Slide the pattern over text one by one
    for i in range(n - m + 1):
        # Check if hash values match
        if p == t:
            # Check if the actual pattern matches
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break

            if match:
                results.append(i)

        # Calculate hash value for next window
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q

            # We might get negative value, convert it to positive
            if t < 0:
                t += q

    return results


def z_algorithm(s: str) -> List[int]:
    """
    Z Algorithm for pattern matching.

    Args:
        s: The input string

    Returns:
        Z array where Z[i] is the length of the longest substring starting from s[i]
        which is also a prefix of s
    """
    n = len(s)
    z = [0] * n

    # Initial window
    left, right = 0, 0

    for i in range(1, n):
        # If i is outside the current window, compute Z[i] naively
        if i > right:
            left = right = i

            # Check if s[left...] matches with s[0...]
            while right < n and s[right] == s[right - left]:
                right += 1

            z[i] = right - left
            right -= 1
        else:
            # We are within the window, copy values
            k = i - left

            # If the value we're copying doesn't hit the window boundary, just copy
            if z[k] < right - i + 1:
                z[i] = z[k]
            else:
                # Otherwise, we need to check beyond the window
                left = i

                while right < n and s[right] == s[right - left]:
                    right += 1

                z[i] = right - left
                right -= 1

    return z


def boyer_moore_search(text: str, pattern: str) -> List[int]:
    """
    Boyer-Moore algorithm for pattern matching in strings.

    This implementation uses the "bad character" rule.

    Args:
        text: The text to search in
        pattern: The pattern to search for

    Returns:
        List of starting indices where the pattern is found in the text
    """
    if not pattern:
        return []

    n = len(text)
    m = len(pattern)
    results = []

    if m > n:
        return results

    # Preprocessing for bad character rule
    # For each character, store its rightmost occurrence in the pattern
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i

    # Search for pattern in text
    s = 0  # Shift of the pattern with respect to text

    while s <= n - m:
        j = m - 1

        # Compare pattern from right to left
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        # Pattern found
        if j < 0:
            results.append(s)
            # Shift pattern beyond the match
            s += 1
        else:
            # Bad character rule: shift pattern by aligning text[s+j] with its
            # rightmost occurrence in pattern, or shift by 1 if not found
            bad_char_shift = j - bad_char.get(text[s + j], -1)
            s += max(1, bad_char_shift)

    return results


def boyer_moore_horspool_search(text: str, pattern: str) -> List[int]:
    """
    Boyer-Moore-Horspool algorithm for pattern matching in strings.

    This is a simplified version of Boyer-Moore that uses only the bad character rule.

    Args:
        text: The text to search in
        pattern: The pattern to search for

    Returns:
        List of starting indices where the pattern is found in the text
    """
    if not pattern:
        return []

    n = len(text)
    m = len(pattern)
    results = []

    if m > n:
        return results

    # Preprocessing: create shift table for all characters
    # For each character, store the shift when a mismatch occurs
    bad_char_table = {}

    for i in range(m - 1):
        bad_char_table[pattern[i]] = m - 1 - i

    # Default shift for characters not in pattern
    default_shift = m

    # Search for pattern in text
    s = 0  # Shift of the pattern with respect to text

    while s <= n - m:
        j = m - 1

        # Compare pattern from right to left
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        # Pattern found
        if j < 0:
            results.append(s)
            s += 1
        else:
            # Shift according to the bad character table
            # or 1 if the character at mismatch is the rightmost pattern character
            char = text[s + m - 1]
            s += bad_char_table.get(char, default_shift)

    return results


def finite_automaton_search(text: str, pattern: str) -> List[int]:
    """
    Finite Automaton algorithm for pattern matching in strings.

    Args:
        text: The text to search in
        pattern: The pattern to search for

    Returns:
        List of starting indices where the pattern is found in the text
    """
    if not pattern:
        return []

    m = len(pattern)
    n = len(text)
    results = []

    # Build transition table
    # transition[state][char] = next_state
    def compute_transition(pattern, m):
        # Default alphabet for ASCII characters
        alphabet = set(pattern)
        alphabet.update(text)

        # Create transition table
        transition = [{} for _ in range(m + 1)]

        for state in range(m + 1):
            for char in alphabet:
                # Find longest prefix that is suffix of "pattern[:state] + char"
                k = min(m, state + 1)

                while k > 0:
                    # Check if pattern[:k] is suffix of "pattern[:state] + char"
                    if pattern[:k - 1] == (pattern[:state] + char)[-k + 1:] if state > 0 else char == pattern[0]:
                        break
                    k -= 1

                transition[state][char] = k

        return transition

    # Compute transition table
    transition = compute_transition(pattern, m)

    # Search for pattern in text
    state = 0
    for i in range(n):
        # Update state based on current character
        if text[i] in transition[state]:
            state = transition[state][text[i]]
        else:
            state = 0

        # Check if we reached accepting state
        if state == m:
            results.append(i - m + 1)

    return results


def sunday_search(text: str, pattern: str) -> List[int]:
    """
    Sunday algorithm for pattern matching in strings.

    Args:
        text: The text to search in
        pattern: The pattern to search for

    Returns:
        List of starting indices where the pattern is found in the text
    """
    if not pattern:
        return []

    n = len(text)
    m = len(pattern)
    results = []

    if m > n:
        return results

    # Preprocessing: create shift table
    # For each character, store its rightmost position from the end of pattern
    shift = {}
    for i in range(m):
        shift[pattern[i]] = m - i

    # Default shift: m + 1 for characters not in pattern
    default_shift = m + 1

    # Search for pattern in text
    i = 0
    while i <= n - m:
        # Check if pattern matches at current position
        j = 0
        while j < m and pattern[j] == text[i + j]:
            j += 1

        # Pattern found
        if j == m:
            results.append(i)

        # Shift based on the character after the current window
        if i + m < n:
            i += shift.get(text[i + m], default_shift)
        else:
            break

    return results