import numpy as np
from numpy import linalg as LA

from scipy.spatial import KDTree

import networkx as nx

import matplotlib.pyplot as plt


class Manifold:
    """
    Represents a 2D differentiable manifold embedded in 3D space,
    discretized by a set of sample points (typically originating from a mesh).
    
    Estimates local differential geometry (tangent and normal spaces),
    builds a connectivity graph between neighboring points,
    and enables geodesic path computation along the manifold.
    """

    __MIN_NEIGHBORHOOD: int = 3

    def __init__(self, points: np.array, k=8) -> None:
        """
        Initializes the manifold from a discrete set of points on the surface.

        Args:
            points (np.array): An (N x 3) array of 3D coordinates sampling the manifold.
            k(int): knn search for PCA.
        """
        self.points = points
        self.k = k
        
        self.tree = KDTree(points)
        self.graph: nx.Graph = nx.Graph()

        self.normal_bundle: np.array = np.zeros((points.shape[0], 3))
        self.tangent_bundle: np.array = np.zeros((points.shape[0], 2, 3))
        self.metric_tensor: np.array = np.zeros((points.shape[0], 2, 2))
        self.__compute_manifold()

    def __get_neighboorhood_data(self, neighborhood: np.array) -> np.array:
        """
        Formats the local neighborhood into a transposed (3 x N) matrix for PCA.

        Args:
            neighborhood (np.array): An (N x 3) array of nearby manifold points.

        Returns:
            np.array: Transposed data suitable for covariance computation (3 x N).
        """
        return neighborhood.T

    def __eigen(self, data: np.array, bias=False) -> tuple[np.array, np.array]:
        """
        Computes and sorts the eigenvalues and eigenvectors of the covariance matrix
        derived from the local neighborhood data.

        The smallest eigenvector corresponds to the surface normal; the remaining two
        form an orthonormal tangent basis.

        Args:
            data (np.array): Transposed neighborhood data (3 x N).
            bias (bool): If True, uses biased covariance estimation.

        Returns:
            tuple:
                - np.array: Sorted eigenvalues (descending).
                - np.array: Corresponding eigenvectors (columns represent directions).
        """
        covariance_matrix = np.cov(data, bias=bias)
        eigenvalues, eigenvectors = LA.eig(covariance_matrix)

        # Order eigenvalues and associated eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def __compute_metric_tensor(self, tangent_space_basis: np.array) -> np.array:
        """
        Computes the Riemannian metric tensor at a point from the given tangent basis vectors.

        Args:
            tangent_space_basis (np.array): A (2 x 3) array of orthonormal tangent vectors.

        Returns:
            np.array: A (2 x 2) symmetric matrix representing the local metric tensor.
        """
        e1, e2 = tangent_space_basis
        return np.array([
            [np.dot(e1, e1), np.dot(e1, e2)],
            [np.dot(e2, e1), np.dot(e2, e2)]
        ])

    def __compute_manifold(self) -> None:
        """
        Builds the internal structure of the manifold:
        - Constructs a connectivity graph using radius-based neighborhoods.
        - Estimates local tangent and normal spaces via PCA at each point.
        - Computes the Riemannian metric tensor at each sample point.
        """
        for i, p in enumerate(self.points):

            distances, indices = self.tree.query(p, k=self.k + 1)
            neighborhood: np.array = self.points[indices]
            
            if len(neighborhood) < self.__MIN_NEIGHBORHOOD:
                continue
            
            # Compute graph
            for idx, j in enumerate(indices):
                if j != i:
                    self.graph.add_edge(i, j, weight=distances[idx])

            # Compute eigenvalues and eigenvectors
            data = self.__get_neighboorhood_data(neighborhood)
            _, eigenvectors = self.__eigen(data)

            # Compute normal and tangent vectors for point p
            normal: np.array = eigenvectors[2]
            self.normal_bundle[i] = normal

            tangent_space_basis: np.array = eigenvectors[:, :2].T
            self.tangent_bundle[i] = tangent_space_basis

            # Compute metric tensor, Christoffel symbols and curvature tensor
            self.metric_tensor[i] = self.__compute_metric_tensor(tangent_space_basis)

    def geodesic(self, start_index: int, end_index: int) -> tuple[np.array, float]:
        """
        Computes a discrete geodesic path between two sample points on the manifold,
        using Dijkstra's algorithm over the connectivity graph.

        Args:
            start_index (int): Index of the source point.
            end_index (int): Index of the target point.

        Returns:
            tuple:
                - np.array: Sequence of vertex indices along the geodesic path.
                - float: Total length of the path.
        """
        shortest_path = nx.shortest_path(self.graph, start_index, end_index, weight="weight")
        total_cost: float = nx.path_weight(self.graph, shortest_path, weight="weight")
        return np.array(shortest_path), total_cost

    def plot_graph(self) -> None:
        """
        Visualizes the manifold's connectivity graph using a 2D layout.
        Useful for debugging and structural insight.
        """
        nx.draw(self.graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
        plt.show()
