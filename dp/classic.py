# dp/classic.py
from typing import List, Dict, Tuple, Set, Optional, Any
import math


def fibonacci_dp(n: int) -> int:
    """
    Calculate the nth Fibonacci number using dynamic programming.

    Args:
        n: The position in the Fibonacci sequence (0-indexed)

    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def coin_change(coins: List[int], amount: int) -> int:
    """
    Find the minimum number of coins needed to make up a given amount.

    Args:
        coins: List of coin denominations
        amount: Target amount

    Returns:
        Minimum number of coins needed, or -1 if impossible
    """
    # Initialize DP array with "infinity"
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins needed to make amount 0

    # Fill the DP array
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1


def knapsack_01(values: List[int], weights: List[int], capacity: int) -> int:
    """
    Solve the 0/1 Knapsack problem.

    Args:
        values: Values of the items
        weights: Weights of the items
        capacity: Maximum weight capacity of the knapsack

    Returns:
        Maximum value that can be obtained
    """
    n = len(values)

    # Create DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                # Either take the item or don't take it
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],  # Take the item
                    dp[i - 1][w]  # Don't take the item
                )
            else:
                # Can't take the item because it's too heavy
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]


def knapsack_01_with_solution(values: List[int], weights: List[int], capacity: int) -> Tuple[int, List[int]]:
    """
    Solve the 0/1 Knapsack problem and reconstruct the solution.

    Args:
        values: Values of the items
        weights: Weights of the items
        capacity: Maximum weight capacity of the knapsack

    Returns:
        Tuple of (maximum value, list of selected item indices)
    """
    n = len(values)

    # Create DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                # Either take the item or don't take it
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],  # Take the item
                    dp[i - 1][w]  # Don't take the item
                )
            else:
                # Can't take the item because it's too heavy
                dp[i][w] = dp[i - 1][w]

    # Reconstruct the solution
    selected_items = []
    w = capacity

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            # Item i was selected
            selected_items.append(i - 1)
            w -= weights[i - 1]

    return dp[n][capacity], list(reversed(selected_items))


def rod_cutting(prices: List[int], n: int) -> int:
    """
    Solve the rod cutting problem.

    Args:
        prices: List of prices where prices[i] is the price of a rod of length i+1
        n: Length of the rod

    Returns:
        Maximum revenue that can be obtained
    """
    # Create DP table
    dp = [0] * (n + 1)

    # Fill the DP table
    for i in range(1, n + 1):
        max_val = float('-inf')
        for j in range(i):
            max_val = max(max_val, prices[j] + dp[i - j - 1])
        dp[i] = max_val

    return dp[n]


def rod_cutting_with_solution(prices: List[int], n: int) -> Tuple[int, List[int]]:
    """
    Solve the rod cutting problem and return the solution.

    Args:
        prices: List of prices where prices[i] is the price of a rod of length i+1
        n: Length of the rod

    Returns:
        Tuple of (maximum revenue, list of cut lengths)
    """
    # Create DP table and cut choice table
    dp = [0] * (n + 1)
    cut = [0] * (n + 1)

    # Fill the DP table
    for i in range(1, n + 1):
        max_val = float('-inf')
        for j in range(i):
            if max_val < prices[j] + dp[i - j - 1]:
                max_val = prices[j] + dp[i - j - 1]
                cut[i] = j + 1
        dp[i] = max_val

    # Reconstruct the solution
    result = []
    remaining = n

    while remaining > 0:
        result.append(cut[remaining])
        remaining -= cut[remaining]

    return dp[n], result


def matrix_chain_multiplication(dims: List[int]) -> int:
    """
    Solve the Matrix Chain Multiplication problem.

    Args:
        dims: Dimensions of matrices. For n matrices, dims has length n+1
              where dims[i-1] x dims[i] is the dimension of matrix i

    Returns:
        Minimum number of scalar multiplications needed
    """
    n = len(dims) - 1  # Number of matrices

    # Create DP table
    # dp[i][j] = minimum cost to multiply matrices i to j
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    # Fill the DP table
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            dp[i][j] = float('inf')

            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                dp[i][j] = min(dp[i][j], cost)

    return dp[1][n]


def max_subarray_sum(nums: List[int]) -> int:
    """
    Find the maximum sum of a contiguous subarray (Kadane's algorithm).

    Args:
        nums: List of integers

    Returns:
        Maximum sum of a contiguous subarray
    """
    if not nums:
        return 0

    current_sum = max_sum = nums[0]

    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum


def max_subarray_sum_indices(nums: List[int]) -> Tuple[int, int, int]:
    """
    Find the maximum sum of a contiguous subarray and its start/end indices.

    Args:
        nums: List of integers

    Returns:
        Tuple of (maximum sum, start index, end index)
    """
    if not nums:
        return 0, -1, -1

    current_sum = max_sum = nums[0]
    start = max_start = max_end = 0

    for i, num in enumerate(nums[1:], 1):
        if num > current_sum + num:
            current_sum = num
            start = i
        else:
            current_sum += num

        if current_sum > max_sum:
            max_sum = current_sum
            max_start = start
            max_end = i

    return max_sum, max_start, max_end


def minimum_partition_difference(nums: List[int]) -> int:
    """
    Partition an array into two subsets such that the difference between the subset sums is minimum.

    Args:
        nums: List of positive integers

    Returns:
        Minimum absolute difference between the sums of two partitions
    """
    if not nums:
        return 0

    total_sum = sum(nums)
    n = len(nums)

    # We want to find a subset with sum as close as possible to total_sum/2
    target = total_sum // 2

    # dp[i][j] = True if a subset of nums[0...i-1] has sum j
    dp = [[False] * (target + 1) for _ in range(n + 1)]

    # Empty subset has sum 0
    for i in range(n + 1):
        dp[i][0] = True

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            # If we can get sum j without the current element
            if j >= nums[i - 1]:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]

    # Find the largest j such that dp[n][j] is true
    j = target
    while j >= 0 and not dp[n][j]:
        j -= 1

    # The difference between the two subset sums is (total_sum - j) - j = total_sum - 2j
    return total_sum - 2 * j


def is_subset_sum(nums: List[int], target_sum: int) -> bool:
    """
    Check if there is a subset of the given set with sum equal to target sum.

    Args:
        nums: List of integers
        target_sum: Target sum to find

    Returns:
        True if there is a subset with the target sum, False otherwise
    """
    n = len(nums)

    # dp[i][j] = True if a subset of nums[0...i-1] has sum j
    dp = [[False] * (target_sum + 1) for _ in range(n + 1)]

    # Empty subset has sum 0
    for i in range(n + 1):
        dp[i][0] = True

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, target_sum + 1):
            if j >= nums[i - 1]:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]

    return dp[n][target_sum]


def count_subset_sum(nums: List[int], target_sum: int) -> int:
    """
    Count the number of subsets of the given set with sum equal to target sum.

    Args:
        nums: List of integers
        target_sum: Target sum to find

    Returns:
        Number of subsets with the target sum
    """
    n = len(nums)

    # dp[i][j] = Number of subsets of nums[0...i-1] with sum j
    dp = [[0] * (target_sum + 1) for _ in range(n + 1)]

    # Empty subset has sum 0
    for i in range(n + 1):
        dp[i][0] = 1

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(target_sum + 1):
            if j >= nums[i - 1]:
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]

    return dp[n][target_sum]


def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    Find the length of the longest strictly increasing subsequence.

    Args:
        nums: List of integers

    Returns:
        Length of the longest increasing subsequence
    """
    if not nums:
        return 0

    n = len(nums)

    # dp[i] = Length of LIS ending at index i
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def longest_common_subsequence_length(s1: str, s2: str) -> int:
    """
    Find the length of the longest common subsequence of two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Length of the longest common subsequence
    """
    m, n = len(s1), len(s2)

    # dp[i][j] = Length of LCS of s1[0...i-1] and s2[0...j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def edit_distance(s1: str, s2: str) -> int:
    """
    Find the minimum number of operations (insert, delete, replace) to convert s1 to s2.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of operations needed
    """
    m, n = len(s1), len(s2)

    # dp[i][j] = Edit distance between s1[0...i-1] and s2[0...j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: empty strings
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

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