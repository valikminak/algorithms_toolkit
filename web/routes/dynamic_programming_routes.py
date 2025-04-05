from flask import Blueprint, jsonify, request

from kosmos.dynamic_programming.classic import knapsack_01_with_solution, matrix_chain_multiplication
from kosmos.dynamic_programming.sequence import longest_common_subsequence

dynamic_programming_routes = Blueprint('dynamic_programming_routes', __name__)


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

        # Generate steps for visualization (simplified)
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
        str1 = data.get('string1', '')
        str2 = data.get('string2', '')

        lcs = longest_common_subsequence(str1, str2)

        # Generate steps for visualization
        steps = []
        m, n = len(str1), len(str2)

        # Initialize DP table
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        # Fill the DP table with steps
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

                # Record the step
                steps.append({
                    'i': i,
                    'j': j,
                    'value': dp[i][j],
                    'match': str1[i - 1] == str2[j - 1],
                    'dp_table': [row[:] for row in dp]
                })

        return jsonify({
            'lcs': lcs,
            'length': dp[m][n],
            'dp_table': dp,
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
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        parenthesis[i][j] = k

                # Record the step
                steps.append({
                    'l': l,
                    'i': i,
                    'j': j,
                    'value': dp[i][j],
                    'dp_table': [row[:] for row in dp],
                    'parenthesis': [row[:] for row in parenthesis]
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
            'min_operations': min_operations,
            'parenthesization': parenthesization,
            'dp_table': dp,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500