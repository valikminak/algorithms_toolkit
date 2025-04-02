import numpy as np
from typing import Optional
from collections import Counter


class LinearRegression:
    """
    Linear Regression implementation from scratch.

    Fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets
    and the targets predicted by the linear approximation.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: Optional[str] = None, lambda_: float = 0.1):
        """
        Initialize Linear Regression model.

        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of gradient descent iterations
            regularization: None, 'l1' for LASSO, or 'l2' for Ridge regression
            lambda_: Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear model using gradient descent.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Self
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model: y = w * X + b
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Add regularization to gradients if specified
            if self.regularization == 'l1':
                # L1 regularization (LASSO)
                dw += self.lambda_ * np.sign(self.weights)
            elif self.regularization == 'l2':
                # L2 regularization (Ridge)
                dw += 2 * self.lambda_ * self.weights

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Args:
            X: Test data of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be fitted before prediction")

        return np.dot(X, self.weights) + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.

        Args:
            X: Test data of shape (n_samples, n_features)
            y: True values of shape (n_samples,)

        Returns:
            R^2 score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


class LogisticRegression:
    """
    Logistic Regression implementation from scratch.

    Logistic regression is a linear model for binary classification
    that uses the logistic function to model the probability of the positive class.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: Optional[str] = None, lambda_: float = 0.1):
        """
        Initialize Logistic Regression model.

        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of gradient descent iterations
            regularization: None, 'l1', or 'l2' regularization
            lambda_: Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Args:
            z: Input to the sigmoid function

        Returns:
            Sigmoid of z
        """
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model using gradient descent.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) with values 0 or 1

        Returns:
            Self
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model + sigmoid
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Add regularization to gradients if specified
            if self.regularization == 'l1':
                dw += self.lambda_ * np.sign(self.weights)
            elif self.regularization == 'l2':
                dw += 2 * self.lambda_ * self.weights

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Args:
            X: Test data of shape (n_samples, n_features)

        Returns:
            Probability estimates of shape (n_samples,)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be fitted before prediction")

        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for X.

        Args:
            X: Test data of shape (n_samples, n_features)
            threshold: Threshold for binary classification

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


class KMeans:
    """
    K-means clustering algorithm implementation from scratch.

    K-means is a centroid-based clustering algorithm that partitions
    a dataset into K distinct, non-overlapping clusters.
    """

    def __init__(self, n_clusters: int = 8, max_iterations: int = 300, tol: float = 1e-4,
                 random_state: Optional[int] = None):
        """
        Initialize K-means clustering.

        Args:
            n_clusters: Number of clusters to form
            max_iterations: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random state for centroid initialization
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centroids.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            Initial centroids of shape (n_clusters, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Select random data points as initial centroids
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Compute k-means clustering.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            Self
        """
        n_samples, n_features = X.shape

        # Initialize centroids
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iterations):
            # Assign samples to closest centroids
            distances = np.zeros((n_samples, self.n_clusters))
            for k in range(self.n_clusters):
                # Compute Euclidean distance to each centroid
                distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)

            # Assign each sample to the closest centroid
            new_labels = np.argmin(distances, axis=1)

            # Check for convergence
            if self.labels_ is not None and np.sum(new_labels != self.labels_) / n_samples < self.tol:
                break

            self.labels_ = new_labels

            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                # If the cluster is empty, keep the current centroid
                if np.sum(self.labels_ == k) > 0:
                    new_centroids[k] = np.mean(X[self.labels_ == k], axis=0)
                else:
                    new_centroids[k] = self.centroids[k]

            # Check for convergence in centroids
            if np.sum(np.linalg.norm(new_centroids - self.centroids, axis=1)) < self.tol * self.n_clusters:
                break

            self.centroids = new_centroids

        # Calculate inertia (sum of squared distances to closest centroid)
        self.inertia_ = 0
        for k in range(self.n_clusters):
            self.inertia_ += np.sum(np.linalg.norm(X[self.labels_ == k] - self.centroids[k], axis=1) ** 2)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.

        Args:
            X: Test data of shape (n_samples, n_features)

        Returns:
            Cluster labels of shape (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")

        # Compute distance to each centroid
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)

        # Return the indices of the closest centroids
        return np.argmin(distances, axis=1)


class DecisionTreeClassifier:
    """
    Decision Tree Classifier implementation from scratch.

    A decision tree recursively partitions the feature space to create a tree structure
    where each internal node represents a decision based on a feature, and each leaf node
    represents a class label.
    """

    class Node:
        """Node in a decision tree."""

        def __init__(self, feature_idx=None, threshold=None, left=None, right=None,
                     value=None, gain=None):
            """
            Initialize a decision tree node.

            Args:
                feature_idx: Feature index to split on
                threshold: Threshold value for the split
                left: Left child node
                right: Right child node
                value: Class prediction if leaf node
                gain: Information gain for the split
            """
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.gain = gain

    def __init__(self, max_depth: int = None, min_samples_split: int = 2,
                 criterion: str = 'gini', random_state: Optional[int] = None):
        """
        Initialize Decision Tree Classifier.

        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split a node
            criterion: Function to measure the quality of a split ('gini' or 'entropy')
            random_state: Random state for feature selection
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.root = None
        self.n_classes = None
        self.feature_importances_ = None

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy for a set of labels.

        Args:
            y: Array of class labels

        Returns:
            Entropy value
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def _calculate_gini(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity for a set of labels.

        Args:
            y: Array of class labels

        Returns:
            Gini impurity value
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def _information_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """
        Calculate information gain for a split.

        Args:
            parent: Parent node labels
            left: Left child node labels
            right: Right child node labels

        Returns:
            Information gain value
        """
        n_left, n_right = len(left), len(right)
        n_parent = n_left + n_right

        if n_parent == 0:
            return 0

        # Calculate impurity based on criterion
        if self.criterion == 'gini':
            impurity_func = self._calculate_gini
        else:  # entropy
            impurity_func = self._calculate_entropy

        parent_impurity = impurity_func(parent)
        left_impurity = impurity_func(left) if n_left > 0 else 0
        right_impurity = impurity_func(right) if n_right > 0 else 0

        # Calculate weighted impurity of children
        weighted_impurity = (n_left / n_parent) * left_impurity + (n_right / n_parent) * right_impurity

        # Return information gain
        return parent_impurity - weighted_impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best split for a node.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)

        Returns:
            Tuple of (best feature index, best threshold, best gain)
        """
        n_samples, n_features = X.shape

        if n_samples <= self.min_samples_split:
            return None, None, 0

        # Set random state for feature selection
        if self.random_state is not None:
            np.random.seed(self.random_state)

        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None

        # Shuffle features to avoid bias
        feature_indices = np.random.permutation(n_features)

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_y = y[left_indices]
                right_y = y[right_indices]

                gain = self._information_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            depth: Current depth in the tree

        Returns:
            Root node of the decision tree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Check stopping criteria
        if (
                self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # Find best split
        best_feature_idx, best_threshold, best_gain = self._best_split(X, y)

        # If no good split found, create a leaf node
        if best_feature_idx is None:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # Create decision node with the best split
        left_indices = np.where(X[:, best_feature_idx] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature_idx] > best_threshold)[0]

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return self.Node(
            feature_idx=best_feature_idx,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            gain=best_gain
        )

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Find the most common label in an array.

        Args:
            y: Array of class labels

        Returns:
            Most common label
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _calculate_feature_importances(self, node, n_samples):
        """
        Calculate feature importances recursively.

        Args:
            node: Current tree node
            n_samples: Total number of samples used to train the tree
        """
        if node.feature_idx is not None:  # Not a leaf node
            # Add feature importance based on node samples and gain
            feature_idx = node.feature_idx
            self.feature_importances_[feature_idx] += node.gain * n_samples

            # Recursively update importances for child nodes
            self._calculate_feature_importances(node.left, n_samples)
            self._calculate_feature_importances(node.right, n_samples)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Build a decision tree classifier from the training set (X, y).

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target classes of shape (n_samples,)

        Returns:
            Self
        """
        self.n_classes = len(np.unique(y))
        _, n_features = X.shape

        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)

        # Build the tree
        self.root = self._build_tree(X, y)

        # Calculate feature importances
        n_samples = X.shape[0]
        self._calculate_feature_importances(self.root, n_samples)

        # Normalize feature importances
        self.feature_importances_ = self.feature_importances_ / self.feature_importances_.sum()

        return self

    def _predict_sample(self, x: np.ndarray, node) -> int:
        """
        Predict the class label for a single sample.

        Args:
            x: Feature vector of shape (n_features,)
            node: Current tree node

        Returns:
            Predicted class label
        """
        # If leaf node, return the prediction
        if node.value is not None:
            return node.value

        # Otherwise, determine which branch to follow
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for X.

        Args:
            X: Test data of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        if self.root is None:
            raise ValueError("Model must be fitted before prediction")

        return np.array([self._predict_sample(x, self.root) for x in X])


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier implementation from scratch.

    KNN is a non-parametric, instance-based learning algorithm that classifies
    new instances based on a majority vote of their k nearest neighbors.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', p: int = 2):
        """
        Initialize KNN Classifier.

        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform' or 'distance')
            p: Power parameter for Minkowski distance (p=1: Manhattan, p=2: Euclidean)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        Fit the k-nearest neighbors classifier.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            Self
        """
        self.X_train = X
        self.y_train = y
        return self

    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Minkowski distance between two points.

        Args:
            x1, x2: Points to calculate distance between

        Returns:
            Minkowski distance
        """
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the input samples.

        Args:
            X: Test data of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before prediction")

        y_pred = np.empty(X.shape[0], dtype=self.y_train.dtype)

        for i, x in enumerate(X):
            # Calculate distances to all training points
            distances = np.array([self._minkowski_distance(x, x_train) for x_train in self.X_train])

            # Find indices of k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_distances = distances[nearest_indices]

            if self.weights == 'uniform':
                # Simple majority vote
                nearest_labels = self.y_train[nearest_indices]
                most_common = Counter(nearest_labels).most_common(1)[0][0]
                y_pred[i] = most_common
            else:  # 'distance' weighting
                # Weighted vote based on inverse distance
                nearest_labels = self.y_train[nearest_indices]

                # Handle case where a training point is exactly at the test point
                nearest_distances[nearest_distances == 0] = 1e-10

                # Calculate weights as inverse of distances
                weights = 1.0 / nearest_distances
                weighted_votes = {}

                for j, label in enumerate(nearest_labels):
                    if label not in weighted_votes:
                        weighted_votes[label] = 0
                    weighted_votes[label] += weights[j]

                # Find the label with the highest weighted vote
                y_pred[i] = max(weighted_votes.items(), key=lambda x: x[1])[0]

        return y_pred