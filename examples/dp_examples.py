from kosmos.dp.classic import (
    fibonacci_dp, coin_change, knapsack_01, knapsack_01_with_solution,
    rod_cutting, rod_cutting_with_solution, matrix_chain_multiplication
)
from kosmos.dp.sequence import (
    longest_increasing_subsequence, longest_common_subsequence,
    longest_common_substring, longest_palindromic_subsequence
)


def classic_dp_examples():
    """Example usage of classic dynamic programming algorithms."""
    # Fibonacci
    n = 10
    print(f"Fibonacci({n}) = {fibonacci_dp(n)}")

    # Coin Change
    coins = [1, 2, 5]
    amount = 11
    print(f"\nMinimum coins to make {amount} from {coins}: {coin_change(coins, amount)}")

    # Knapsack
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    print("\nKnapsack Problem:")
    max_value = knapsack_01(values, weights, capacity)
    print(f"Maximum value: {max_value}")

    max_value, selected_items = knapsack_01_with_solution(values, weights, capacity)
    print(f"Selected items: {selected_items} with value: {max_value}")

    # Rod Cutting
    prices = [1, 5, 8, 9, 10, 17, 17, 20]
    rod_length = 8

    print("\nRod Cutting Problem:")
    max_revenue = rod_cutting(prices, rod_length)
    print(f"Maximum revenue: {max_revenue}")

    max_revenue, cuts = rod_cutting_with_solution(prices, rod_length)
    print(f"Optimal cuts: {cuts} with revenue: {max_revenue}")

    # Matrix Chain Multiplication
    dimensions = [10, 30, 5, 60]  # 3 matrices: 10x30, 30x5, 5x60
    min_ops = matrix_chain_multiplication(dimensions)
    print(f"\nMinimum operations for matrix multiplication: {min_ops}")


def sequence_dp_examples():
    """Example usage of sequence dynamic programming algorithms."""
    # Longest Increasing Subsequence
    nums = [10, 22, 9, 33, 21, 50, 41, 60, 80]
    print(f"Longest Increasing Subsequence: {longest_increasing_subsequence(nums)}")

    # Longest Common Subsequence
    s1 = "ABCBDAB"
    s2 = "BDCABA"
    print(f"\nLongest Common Subsequence of '{s1}' and '{s2}': {longest_common_subsequence(s1, s2)}")

    # Longest Common Substring
    s1 = "www.example.com"
    s2 = "www.example.org"
    print(f"\nLongest Common Substring of '{s1}' and '{s2}': {longest_common_substring(s1, s2)}")

    # Longest Palindromic Subsequence
    s = "BBABCBCAB"
    print(f"\nLongest Palindromic Subsequence of '{s}': {longest_palindromic_subsequence(s)}")


def run_all_examples():
    """Run all dynamic programming examples."""
    print("=" * 50)
    print("CLASSIC DYNAMIC PROGRAMMING EXAMPLES")
    print("=" * 50)
    classic_dp_examples()

    print("\n" + "=" * 50)
    print("SEQUENCE DYNAMIC PROGRAMMING EXAMPLES")
    print("=" * 50)
    sequence_dp_examples()


if __name__ == "__main__":
    run_all_examples()