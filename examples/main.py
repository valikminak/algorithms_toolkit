import os
import sys

# Add the parent directory to system path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.graph_examples import run_all_examples as run_graph_examples
from examples.tree_examples import run_all_examples as run_tree_examples
from examples.string_examples import run_all_examples as run_string_examples
from examples.dp_examples import run_all_examples as run_dp_examples
from examples.sorting_searching_examples import run_all_examples as run_sorting_searching_examples
from examples.geometry_examples import run_all_examples as run_geometry_examples
from examples.advanced_examples import run_all_examples as run_advanced_examples


def main():
    """Main function to run all examples."""
    print("=" * 60)
    print("ADVANCED ALGORITHMS AND DATA STRUCTURES TOOLKIT EXAMPLES")
    print("=" * 60)

    # Menu for examples
    menu = {
        1: ("Graph Algorithms", run_graph_examples),
        2: ("Tree Algorithms", run_tree_examples),
        3: ("String Algorithms", run_string_examples),
        4: ("Dynamic Programming", run_dp_examples),
        5: ("Sorting and Searching", run_sorting_searching_examples),
        6: ("Computational Geometry", run_geometry_examples),
        7: ("Advanced Algorithms", run_advanced_examples),
        8: ("Run All Examples", run_all_examples),
        0: ("Exit", None)
    }

    while True:
        print("\nSelect an example category to run:")
        for key, (name, _) in menu.items():
            print(f"{key}. {name}")

        try:
            choice = int(input("\nEnter your choice (0-8): "))

            if choice == 0:
                print("Exiting...")
                break
            elif choice in menu:
                name, func = menu[choice]
                print(f"\nRunning {name} Examples...\n")

                if choice == 8:
                    run_all_examples()
                else:
                    func()
            else:
                print("Invalid choice. Please enter a number between 0 and 8.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def run_all_examples():
    """Run all examples from all categories."""
    print("\n" + "=" * 60)
    print("GRAPH ALGORITHMS")
    print("=" * 60)
    run_graph_examples()

    print("\n" + "=" * 60)
    print("TREE ALGORITHMS")
    print("=" * 60)
    run_tree_examples()

    print("\n" + "=" * 60)
    print("STRING ALGORITHMS")
    print("=" * 60)
    run_string_examples()

    print("\n" + "=" * 60)
    print("DYNAMIC PROGRAMMING")
    print("=" * 60)
    run_dp_examples()

    print("\n" + "=" * 60)
    print("SORTING AND SEARCHING")
    print("=" * 60)
    run_sorting_searching_examples()

    print("\n" + "=" * 60)
    print("COMPUTATIONAL GEOMETRY")
    print("=" * 60)
    run_geometry_examples()

    print("\n" + "=" * 60)
    print("ADVANCED ALGORITHMS")
    print("=" * 60)
    run_advanced_examples()


if __name__ == "__main__":
    main()