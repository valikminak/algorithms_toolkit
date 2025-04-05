from flask import Blueprint, jsonify, request

from kosmos.dynamic_programming.classic import knapsack_01_with_solution, matrix_chain_multiplication, fibonacci_dp
from kosmos.dynamic_programming.sequence import longest_common_subsequence, longest_increasing_subsequence

dynamic_programming_routes = Blueprint('dynamic_programming_routes', __name__)


@dynamic_programming_routes.route('/api/dp/algorithms')
def get_dp_algorithms():
    """Return available dynamic programming algorithms"""
    algorithms = [
        {"id": "fibonacci", "name": "Fibonacci", "complexity": "O(n)"},
        {"id": "knapsack_01", "name": "0/1 Knapsack", "complexity": "O(nW)"},
        {"id": "lcs", "name": "Longest Common Subsequence", "complexity": "O(mn)"},
        {"id": "lis", "name": "Longest Increasing Subsequence", "complexity": "O(n²)"},
        {"id": "matrix_chain", "name": "Matrix Chain Multiplication", "complexity": "O(n³)"}
    ]
    return jsonify(algorithms)


@dynamic_programming_routes.route('/api/dp/fibonacci', methods=['POST'])
def fibonacci():
    """
    Fibonacci dynamic programming implementation
    Input: JSON with n value
    Output: JSON with fibonacci number, table, and steps
    """
    try:
        data = request.get_json()
        n = data.get('n', 10)

        if n < 0:
            return jsonify({'error': 'n must be non-negative'}), 400

        if n > 100:
            return jsonify({'error': 'n too large, maximum allowed is 100'}), 400

        # Calculate fibonacci
        result = fibonacci_dp(n)

        # Generate steps for visualization
        steps = []
        dp_table = [0] * (n + 1)

        # Base cases
        if n >= 0:
            dp_table[0] = 0
            steps.append({
                'dp_table': dp_table.copy(),
                'current_index': 0,
                'info': 'Base case: F(0) = 0'
            })

        if n >= 1:
            dp_table[1] = 1
            steps.append({
                'dp_table': dp_table.copy(),
                'current_index': 1,
                'info': 'Base case: F(1) = 1'
            })

        # Fill DP table
        for i in range(2, n + 1):
            dp_table[i] = dp_table[i - 1] + dp_table[i - 2]
            steps.append({
                'dp_table': dp_table.copy(),
                'current_index': i,
                'info': f'F({i}) = F({i - 1}) + F({i - 2}) = {dp_table[i - 1]} + {dp_table[i - 2]} = {dp_table[i]}'
            })

        return jsonify({
            'algorithm': 'fibonacci',
            'category': 'dp',
            'n': n,
            'result': result,
            'dp_table': dp_table,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@dynamic_programming_routes.route('/api/dp/knapsack', methods=['POST'])
def knapsack():
    """
    0-1 Knapsack algorithm implementation
    Input: JSON with weights, values, and capacity
    Output: JSON with solution, table, and steps
    """
    try:
        data = request.get_json()
        weights = data.get('weights', [])
        values = data.get('values', [])
        capacity = data.get('capacity', 0)

        if len(weights) != len(values):
            return jsonify({'error': 'Weights and values arrays must have the same length'}), 400

        max_value, selected_items = knapsack_01_with_solution(values, weights, capacity)

        # Generate steps for visualization
        steps = []
        n = len(weights)

        # Create DP table
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        # Fill the DP table with steps
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                # Don't include item i
                dp[i][w] = dp[i - 1][w]

                # If current item can fit, decide whether to include it
                if weights[i - 1] <= w:
                    # Max of (not including item, including item)
                    dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

                # Record the step for visualization
                steps.append({
                    'i': i,
                    'w': w,
                    'value': dp[i][w],
                    'included': dp[i][w] != dp[i - 1][w],
                    'dp_table': [row[:] for row in dp]
                })

        return jsonify({
            'algorithm': 'knapsack_01',
            'category': 'dp',
            'weights': weights,
            'values': values,
            'capacity': capacity,
            'max_value': max_value,
            'selected_items': selected_items,
            'dp_table': dp,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@dynamic_programming_routes.route('/api/dp/lcs', methods=['POST'])
def longest_common_subsequence_route():
    """
    Longest Common Subsequence algorithm implementation
    Input: JSON with two strings
    Output: JSON with LCS, table, and steps
    """
    try:
        data = request.get_json()
        string1 = data.get('string1', '')
        string2 = data.get('string2', '')

        lcs = longest_common_subsequence(string1, string2)

        # Generate steps for visualization
        steps = []
        m, n = len(string1), len(string2)

        # Initialize DP table
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        # Fill the DP table with steps
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if string1[i - 1] == string2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    steps.append({
                        'i': i,
                        'j': j,
                        'value': dp[i][j],
                        'match': True,
                        'dp_table': [row[:] for row in dp],
                        'info': f"Match: '{string1[i - 1]}' at positions string1[{i - 1}] and string2[{j - 1}]"
                    })
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    steps.append({
                        'i': i,
                        'j': j,
                        'value': dp[i][j],
                        'match': False,
                        'dp_table': [row[:] for row in dp],
                        'info': f"No match between '{string1[i - 1]}' and '{string2[j - 1]}'"
                    })

        # Reconstruct the LCS
        i, j = m, n
        lcs_chars = []

        while i > 0 and j > 0:
            if string1[i - 1] == string2[j - 1]:
                lcs_chars.append(string1[i - 1])
                steps.append({
                    'i': i,
                    'j': j,
                    'dp_table': [row[:] for row in dp],
                    'backtrack': True,
                    'lcs_so_far': ''.join(reversed(lcs_chars)),
                    'info': f"Backtracking: Add '{string1[i - 1]}' to LCS"
                })
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                steps.append({
                    'i': i,
                    'j': j,
                    'dp_table': [row[:] for row in dp],
                    'backtrack': True,
                    'lcs_so_far': ''.join(reversed(lcs_chars)),
                    'info': "Backtracking: Move up (i-1, j)"
                })
                i -= 1
            else:
                steps.append({
                    'i': i,
                    'j': j,
                    'dp_table': [row[:] for row in dp],
                    'backtrack': True,
                    'lcs_so_far': ''.join(reversed(lcs_chars)),
                    'info': "Backtracking: Move left (i, j-1)"
                })
                j -= 1

        return jsonify({
            'algorithm': 'lcs',
            'category': 'dp',
            'string1': string1,
            'string2': string2,
            'lcs': lcs,
            'length': dp[m][n],
            'dp_table': dp,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@dynamic_programming_routes.route('/api/dp/lis', methods=['POST'])
def longest_increasing_subsequence_route():
    """
    Longest Increasing Subsequence algorithm implementation
    Input: JSON with an array of numbers
    Output: JSON with LIS, table, and steps
    """
    try:
        data = request.get_json()
        nums = data.get('nums', [])

        lis = longest_increasing_subsequence(nums)

        # Generate steps for visualization
        steps = []
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

                    steps.append({
                        'i': i,
                        'j': j,
                        'dp': dp.copy(),
                        'prev': prev.copy(),
                        'info': f"Found new LIS: {nums[i]} > {nums[j]}, length = {dp[i]}"
                    })
                else:
                    steps.append({
                        'i': i,
                        'j': j,
                        'dp': dp.copy(),
                        'prev': prev.copy(),
                        'info': f"No update: {'nums[i] <= nums[j]' if nums[i] <= nums[j] else 'Not a longer subsequence'}"
                    })

        # Find the index with maximum LIS length
        max_length = max(dp)
        max_index = dp.index(max_length)

        # Reconstruct the LIS
        lis_indices = []
        while max_index != -1:
            lis_indices.append(max_index)
            max_index = prev[max_index]

            steps.append({
                'backtrack': True,
                'lis_indices': lis_indices.copy(),
                'dp': dp.copy(),
                'prev': prev.copy(),
                'info': f"Backtracking: Current LIS indices = {lis_indices}"
            })

        lis_indices.reverse()
        lis_values = [nums[i] for i in lis_indices]

        return jsonify({
            'algorithm': 'lis',
            'category': 'dp',
            'nums': nums,
            'lis': lis,
            'lis_indices': lis_indices,
            'lis_values': lis_values,
            'dp': dp,
            'prev': prev,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@dynamic_programming_routes.route('/api/dp/matrix_chain', methods=['POST'])
def matrix_chain_route():
    """
    Matrix Chain Multiplication algorithm implementation
    Input: JSON with array of matrix dimensions
    Output: JSON with optimal solution, table, and steps
    """
    try:
        data = request.get_json()
        dimensions = data.get('dimensions', [])

        if len(dimensions) < 2:
            return jsonify({'error': 'At least 2 dimensions required'}), 400

        min_operations = matrix_chain_multiplication(dimensions)

        # Generate steps for visualization
        steps = []
        n = len(dimensions) - 1  # Number of matrices

        # Initialize DP table
        dp = [[0 for _ in range(n)] for _ in range(n)]
        parenthesis = [[0 for _ in range(n)] for _ in range(n)]

        # Fill the DP table with steps
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                dp[i][j] = float('inf')

                steps.append({
                    'l': l,
                    'i': i,
                    'j': j,
                    'value': None,
                    'dp_table': [row[:] for row in dp],
                    'parenthesis': [row[:] for row in parenthesis],
                    'info': f"Computing optimal cost for chain A{i}...A{j}"
                })

                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]

                    steps.append({
                        'l': l,
                        'i': i,
                        'j': j,
                        'k': k,
                        'cost': cost,
                        'dp_table': [row[:] for row in dp],
                        'parenthesis': [row[:] for row in parenthesis],
                        'info': f"Testing split at k={k}: {dp[i][k]} + {dp[k + 1][j]} + {dimensions[i]}*{dimensions[k + 1]}*{dimensions[j + 1]} = {cost}"
                    })

                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parenthesis[i][j] = k

                        steps.append({
                            'l': l,
                            'i': i,
                            'j': j,
                            'k': k,
                            'update': True,
                            'value': cost,
                            'dp_table': [row[:] for row in dp],
                            'parenthesis': [row[:] for row in parenthesis],
                            'info': f"New minimum cost: {cost} with split at k={k}"
                        })

        # Generate optimal parenthesization
        def get_parenthesization(i, j):
            if i == j:
                return f"A{i + 1}"
            else:
                k = parenthesis[i][j]
                return f"({get_parenthesization(i, k)}{get_parenthesization(k + 1, j)})"

        parenthesization = get_parenthesization(0, n - 1)

        return jsonify({
            'algorithm': 'matrix_chain',
            'category': 'dp',
            'dimensions': dimensions,
            'min_operations': min_operations,
            'parenthesization': parenthesization,
            'dp_table': dp,
            'parenthesis': parenthesis,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500