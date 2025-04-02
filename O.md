# Sorting Algorithms: Quick Reference

| Algorithm      | Time Complexity (Average) | Time Complexity (Worst) | Space Complexity | Stable | In-Place | Adaptive | Best For |
|----------------|---------------------------|-------------------------|------------------|--------|----------|----------|----------|
| Bubble Sort    | O(n²)                    | O(n²)                   | O(1)             | ✅     | ✅       | ✅       | Small arrays, nearly sorted data |
| Selection Sort | O(n²)                    | O(n²)                   | O(1)             | ❌     | ✅       | ❌       | Small arrays, minimizing writes |
| Insertion Sort | O(n²)                    | O(n²)                   | O(1)             | ✅     | ✅       | ✅       | Small arrays, online sorting, nearly sorted data |
| Merge Sort     | O(n log n)               | O(n log n)              | O(n)             | ✅     | ❌       | ❌       | Guaranteed performance, linked lists |
| Quick Sort     | O(n log n)               | O(n²)                   | O(log n)         | ❌     | ✅       | ❌       | Arrays, general purpose |
| Heap Sort      | O(n log n)               | O(n log n)              | O(1)             | ❌     | ✅       | ❌       | Memory constraints, guaranteed performance |
| Counting Sort  | O(n + k)                 | O(n + k)                | O(n + k)         | ✅     | ❌       | ❌       | Small range of integers |
| Radix Sort     | O(nk)                    | O(nk)                   | O(n + k)         | ✅     | ❌       | ❌       | Fixed-length integers or strings |

## Key Terms
- **Stable**: Preserves relative order of equal elements
- **In-Place**: Requires O(1) extra space
- **Adaptive**: Performance improves for partially sorted data

## Visual Comparison
Different sorting algorithms perform differently based on:
- Input size
- Initial order of elements
- Distribution of keys
- Available memory

## When to Choose Each Algorithm
- **Bubble Sort**: Educational purposes, very small arrays
- **Insertion Sort**: Small arrays (<20 elements), nearly sorted data, online sorting
- **Merge Sort**: Stable sorting needed, external sorting, linked lists
- **Quick Sort**: General purpose, in-memory sorting, large arrays
- **Heap Sort**: Memory-constrained environments, guaranteed O(n log n)
- **Counting/Radix Sort**: When keys are in a small range or fixed length