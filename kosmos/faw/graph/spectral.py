import numpy as np
from typing import List, Dict
from scipy import linalg


def laplacian_matrix(adjacency_matrix: np.ndarray, normalized: bool = False) -> np.ndarray:
    """
    Compute the Laplacian matrix of a graph from its adjacency matrix.

    The Laplacian matrix is defined as L = D - A, where D is the degree matrix
    and A is the adjacency matrix. For the normalized Laplacian, L = I - D^(-1/2)AD^(-1/2).

    Args:
        adjacency_matrix: Graph adjacency matrix
        normalized: Whether to compute the normalized Laplacian

    Returns:
        Laplacian matrix
    """
    # Ensure the matrix is symmetric (undirected graph)
    if not np.allclose(adjacency_matrix, adjacency_matrix.T):
        raise ValueError("Adjacency matrix must be symmetric (undirected graph)")

    # Get number of vertices
    n = adjacency_matrix.shape[0]

    # Compute the degree matrix
    degrees = np.sum(adjacency_matrix, axis=1)
    degree_matrix = np.diag(degrees)

    if normalized:
        # For isolated vertices (degree=0), we use a degree of 1 to avoid division by zero
        degrees_fixed = np.where(degrees > 0, degrees, 1)

        # Compute D^(-1/2)
        degree_matrix_invsqrt = np.diag(1.0 / np.sqrt(degrees_fixed))

        # Compute the normalized Laplacian: I - D^(-1/2)AD^(-1/2)
        laplacian = np.eye(n) - degree_matrix_invsqrt @ adjacency_matrix @ degree_matrix_invsqrt
    else:
        # Compute the standard Laplacian: D - A
        laplacian = degree_matrix - adjacency_matrix

    return laplacian


def spectral_clustering(adjacency_matrix: np.ndarray, n_clusters: int) -> List[int]:
    """
    Perform spectral clustering on a graph.

    Spectral clustering uses the eigenvectors of the graph Laplacian to
    embed the vertices in a lower dimensional space, followed by k-means
    clustering to partition the vertices.

    Args:
        adjacency_matrix: Graph adjacency matrix
        n_clusters: Number of clusters

    Returns:
        List of cluster labels for each vertex
    """
    # Compute normalized Laplacian
    laplacian = laplacian_matrix(adjacency_matrix, normalized=True)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eigh(laplacian)

    # Get the eigenvectors corresponding to the k smallest non-zero eigenvalues
    # The smallest eigenvalue should be close to zero, which corresponds to the constant vector
    indices = np.argsort(eigenvalues)[1:n_clusters + 1]  # Skip the smallest eigenvalue
    features = eigenvectors[:, indices]

    # Normalize the rows of the features matrix
    row_norms = np.sqrt(np.sum(features ** 2, axis=1))
    features = features / row_norms[:, np.newaxis]

    # Apply k-means clustering to the features
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(features)

    return clusters


def pagerank(adjacency_matrix: np.ndarray, damping_factor: float = 0.85,
             max_iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Compute PageRank scores for vertices in a directed graph.

    PageRank is an algorithm used by Google to rank web pages in search results.
    It models the behavior of a random surfer who either follows a link or
    jumps to a random page with probability (1-damping_factor).

    Args:
        adjacency_matrix: Directed graph adjacency matrix
        damping_factor: Probability of following a link (typically 0.85)
        max_iterations: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Array of PageRank scores for each vertex
    """
    n = adjacency_matrix.shape[0]

    # Convert adjacency matrix to transition probability matrix
    # by normalizing columns to sum to 1
    column_sums = np.sum(adjacency_matrix, axis=0)

    # Handle columns with all zeros (vertices with no outgoing edges)
    column_sums = np.where(column_sums > 0, column_sums, 1)

    # P[i, j] is the probability of moving from vertex j to vertex i
    transition_matrix = adjacency_matrix / column_sums

    # Initialize PageRank scores to uniform distribution
    pr = np.ones(n) / n

    # Power iteration method to compute PageRank
    for _ in range(max_iterations):
        pr_next = (1 - damping_factor) / n + damping_factor * (transition_matrix @ pr)

        # Check convergence
        if np.linalg.norm(pr_next - pr, 1) < tol:
            return pr_next

        pr = pr_next

    return pr


def graph_centrality_metrics(adjacency_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute various centrality metrics for vertices in a graph.

    Centrality metrics measure the importance or influence of vertices in a graph.

    Args:
        adjacency_matrix: Graph adjacency matrix

    Returns:
        Dictionary mapping metric names to arrays of scores for each vertex
    """
    n = adjacency_matrix.shape[0]
    results = {}

    # Degree centrality
    degree = np.sum(adjacency_matrix, axis=1)
    results['degree_centrality'] = degree / (n - 1)  # Normalize

    # Closeness centrality
    # First compute the shortest path distances using Floyd-Warshall
    distances = np.zeros_like(adjacency_matrix, dtype=float)
    distances[adjacency_matrix > 0] = 1  # Initialize with edge weights
    distances[adjacency_matrix == 0] = np.inf
    np.fill_diagonal(distances, 0)  # Diagonal should be zero

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distances[i, j] > distances[i, k] + distances[k, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]

    # Compute closeness for each vertex
    closeness = np.zeros(n)
    for i in range(n):
        # Sum of finite distances (exclude unreachable vertices)
        finite_sum = np.sum(distances[i, :] < np.inf)
        if finite_sum > 1:  # If vertex can reach others
            closeness[i] = (finite_sum - 1) / np.sum(distances[i, distances[i, :] < np.inf])

    results['closeness_centrality'] = closeness

    # Betweenness centrality (simple approximation, not the full algorithm)
    betweenness = np.zeros(n)

    # Compute the number of shortest paths going through each vertex
    for s in range(n):
        for t in range(s + 1, n):
            # Find all vertices on shortest paths from s to t
            path_vertices = set()
            for v in range(n):
                if v != s and v != t:
                    if distances[s, v] + distances[v, t] == distances[s, t]:
                        path_vertices.add(v)

            # Increment betweenness for vertices on shortest paths
            for v in path_vertices:
                betweenness[v] += 1

    # Normalize betweenness
    betweenness = betweenness / ((n - 1) * (n - 2) / 2)
    results['betweenness_centrality'] = betweenness

    # Eigenvector centrality
    eigenvalues, eigenvectors = linalg.eigh(adjacency_matrix)
    idx = np.argmax(eigenvalues)  # Index of largest eigenvalue
    eigenvector_centrality = np.abs(eigenvectors[:, idx])
    eigenvector_centrality = eigenvector_centrality / np.linalg.norm(eigenvector_centrality, 1)
    results['eigenvector_centrality'] = eigenvector_centrality

    # PageRank centrality
    pagerank_centrality = pagerank(adjacency_matrix)
    results['pagerank_centrality'] = pagerank_centrality

    return results


def community_detection_label_propagation(adjacency_matrix: np.ndarray,
                                          max_iterations: int = 100) -> List[int]:
    """
    Detect communities in a graph using the Label Propagation algorithm.

    The Label Propagation algorithm iteratively assigns each vertex the label
    that most of its neighbors have, until convergence.

    Args:
        adjacency_matrix: Graph adjacency matrix
        max_iterations: Maximum number of iterations

    Returns:
        List of community labels for each vertex
    """
    n = adjacency_matrix.shape[0]

    # Initialize labels (each vertex has its own community)
    labels = np.arange(n)

    # Shuffle the vertices to avoid biases
    vertices = np.arange(n)

    # Label propagation
    for _ in range(max_iterations):
        np.random.shuffle(vertices)
        old_labels = labels.copy()

        # Update labels for each vertex
        for v in vertices:
            # Get neighbors and their labels
            neighbors = np.where(adjacency_matrix[v, :] > 0)[0]
            if len(neighbors) == 0:
                continue

            neighbor_labels = labels[neighbors]

            # Count label occurrences
            label_counts = {}
            for label in neighbor_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            # Find the most common label(s)
            max_count = max(label_counts.values()) if label_counts else 0
            most_common = [label for label, count in label_counts.items() if count == max_count]

            # If there are multiple most common labels, choose randomly
            if most_common:
                labels[v] = np.random.choice(most_common)

        # Check for convergence
        if np.array_equal(labels, old_labels):
            break

    # Relabel communities from 0 to num_communities-1
    unique_labels = np.unique(labels)
    new_labels = np.zeros(n, dtype=int)
    for i, label in enumerate(unique_labels):
        new_labels[labels == label] = i

    return new_labels


def spectral_embedding(adjacency_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Compute a spectral embedding of a graph in low-dimensional space.

    Spectral embedding uses the eigenvectors of the graph Laplacian to
    map vertices into a low-dimensional space where distances represent
    structural similarities.

    Args:
        adjacency_matrix: Graph adjacency matrix
        n_components: Number of dimensions in the embedding

    Returns:
        Matrix of shape (n_vertices, n_components) containing the embedding
    """
    # Compute normalized Laplacian
    laplacian = laplacian_matrix(adjacency_matrix, normalized=True)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eigh(laplacian)

    # Get the eigenvectors corresponding to the smallest non-zero eigenvalues
    indices = np.argsort(eigenvalues)[1:n_components + 1]  # Skip the smallest eigenvalue
    embedding = eigenvectors[:, indices]

    # Scale the embedding by the square root of the eigenvalues for better preservation of distances
    embedding = embedding * np.sqrt(1.0 / eigenvalues[indices])

    return embedding


def diffusion_map(adjacency_matrix: np.ndarray, n_components: int = 2,
                  alpha: float = 0.5, t: float = 1.0) -> np.ndarray:
    """
    Compute a diffusion map embedding of a graph.

    Diffusion maps use the eigenvectors of a diffusion process on the graph
    to map vertices into a low-dimensional space where distances represent
    diffusion distances.

    Args:
        adjacency_matrix: Graph adjacency matrix
        n_components: Number of dimensions in the embedding
        alpha: Diffusion parameter (0 <= alpha <= 1)
        t: Diffusion time

    Returns:
        Matrix of shape (n_vertices, n_components) containing the embedding
    """
    adjacency_matrix.shape[0]

    # Ensure symmetric matrix
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2

    # Compute the degree matrix
    degrees = np.sum(adjacency_matrix, axis=1)
    degree_matrix_inv = np.diag(1.0 / np.where(degrees > 0, degrees, 1))

    # Create Markov matrix (normalized adjacency matrix)
    markov_matrix = degree_matrix_inv @ adjacency_matrix

    # Create the diffusion matrix with alpha normalization
    if alpha > 0:
        degree_matrix_alpha = np.diag(degrees ** (-alpha))
        kernel = degree_matrix_alpha @ adjacency_matrix @ degree_matrix_alpha
        degrees_kernel = np.sum(kernel, axis=1)
        degree_matrix_kernel_inv = np.diag(1.0 / np.where(degrees_kernel > 0, degrees_kernel, 1))
        markov_matrix = degree_matrix_kernel_inv @ kernel

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eigh(markov_matrix)

    # Sort by eigenvalues in descending order (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip the first eigenvector (corresponds to eigenvalue 1)
    embedding_indices = np.arange(1, n_components + 1)

    # Scale the eigenvectors by eigenvalues to the power of t (diffusion time)
    embedding = eigenvectors[:, embedding_indices] * eigenvalues[embedding_indices] ** t

    return embedding