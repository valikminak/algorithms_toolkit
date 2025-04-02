from algorithms_toolkit.graph.base import Graph
from algorithms_toolkit.graph.traversal import breadth_first_search, depth_first_search, topological_sort
from algorithms_toolkit.graph.shortest_path import dijkstra, bellman_ford, floyd_warshall
from algorithms_toolkit.graph.mst import kruskal_mst, prim_mst
from algorithms_toolkit.graph.connectivity import tarjan_scc, articulation_points, bridges
from algorithms_toolkit.graph.flow import ford_fulkerson, edmonds_karp, min_cut
from algorithms_toolkit.graph.coloring import greedy_coloring
from algorithms_toolkit.graph.bipartite import is_bipartite, maximum_bipartite_matching
from algorithms_toolkit.utils.visualization import visualize_graph


def create_sample_graph():
    """Create a sample graph for demonstration."""
    g = Graph(directed=False, weighted=True)

    # Add vertices
    for i in range(6):
        g.add_vertex(i)

    # Add edges
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 2)
    g.add_edge(1, 2, 5)
    g.add_edge(1, 3, 10)
    g.add_edge(2, 3, 3)
    g.add_edge(2, 4, 8)
    g.add_edge(3, 4, 2)
    g.add_edge(3, 5, 7)
    g.add_edge(4, 5, 6)

    return g


def traversal_example():
    """Example usage of graph traversal algorithms."""
    g = create_sample_graph()
    print("Graph:")
    print(g)

    # BFS
    print("\nBFS starting from vertex 0:")
    bfs_result = breadth_first_search(g, 0)
    for vertex, parent in bfs_result.items():
        print(f"Vertex {vertex}: parent = {parent if parent is not None else 'None (root)'}")

    # DFS
    print("\nDFS starting from vertex 0:")
    dfs_result = depth_first_search(g, 0)
    for vertex, (discover, finish, parent) in dfs_result.items():
        print(
            f"Vertex {vertex}: discovery={discover}, finish={finish}, parent={parent if parent is not None else 'None'}")

    # Create a directed graph for topological sort
    dag = Graph(directed=True)
    dag.add_edge(5, 2)
    dag.add_edge(5, 0)
    dag.add_edge(4, 0)
    dag.add_edge(4, 1)
    dag.add_edge(2, 3)
    dag.add_edge(3, 1)

    print("\nTopological Sort of a DAG:")
    topo_order = topological_sort(dag)
    print(f"Topological order: {topo_order}")

    # Visualize the graph
    # visualize_graph(g, layout='spring')


def shortest_path_example():
    """Example usage of shortest path algorithms."""
    g = create_sample_graph()

    # Dijkstra's algorithm
    print("Dijkstra's algorithm from vertex 0:")
    distances, predecessors = dijkstra(g, 0)
    for vertex, distance in distances.items():
        print(f"Distance to vertex {vertex}: {distance}")

    # Bellman-Ford algorithm
    print("\nBellman-Ford algorithm from vertex 0:")
    distances, predecessors, no_negative_cycle = bellman_ford(g, 0)
    print(f"Negative cycle detected: {not no_negative_cycle}")
    for vertex, distance in distances.items():
        print(f"Distance to vertex {vertex}: {distance}")

    # Floyd-Warshall algorithm
    print("\nFloyd-Warshall algorithm:")
    distances, next_vertices = floyd_warshall(g)
    for u in g.vertices:
        for v in g.vertices:
            if u != v:
                print(f"Shortest distance from {u} to {v}: {distances[(u, v)]}")


def mst_example():
    """Example usage of minimum spanning tree algorithms."""
    g = create_sample_graph()

    # Kruskal's algorithm
    print("Kruskal's MST:")
    mst_edges = kruskal_mst(g)
    total_weight = sum(weight for _, _, weight in mst_edges)
    print(f"Edges in the MST: {mst_edges}")
    print(f"Total weight: {total_weight}")

    # Prim's algorithm
    print("\nPrim's MST:")
    mst_edges = prim_mst(g)
    total_weight = sum(weight for _, _, weight in mst_edges)
    print(f"Edges in the MST: {mst_edges}")
    print(f"Total weight: {total_weight}")


def run_all_examples():
    """Run all graph algorithm examples."""
    print("=" * 50)
    print("GRAPH TRAVERSAL EXAMPLES")
    print("=" * 50)
    traversal_example()

    print("\n" + "=" * 50)
    print("SHORTEST PATH EXAMPLES")
    print("=" * 50)
    shortest_path_example()

    print("\n" + "=" * 50)
    print("MINIMUM SPANNING TREE EXAMPLES")
    print("=" * 50)
    mst_example()


if __name__ == "__main__":
    run_all_examples()