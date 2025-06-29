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
        
        # Compute manifold
        self.tree = KDTree(points)
        self.graph: nx.Graph = nx.Graph()

        self.normal_bundle: np.array = np.zeros((points.shape[0], 3))
        self.normalized_normal_bundle: np.array = np.zeros((points.shape[0], 3))
        self.tangent_bundle: np.array = np.zeros((points.shape[0], 2, 3))
        self.metric_tensor: np.array = np.zeros((points.shape[0], 2, 2))
        self.metric_tensor_inv: np.array = np.zeros(self.metric_tensor.shape)
        self.__compute_manifold()
        
        # Compute curvature
        self.metric_tensor_derivatives: np.array = np.zeros((self.points.shape[0], 2, 2, 2))
        self.christoffel_symbols: np.array = np.zeros((self.points.shape[0], 2, 2, 2))
        self.christoffel_symbols_derivatives: np.array = np.zeros((self.points.shape[0], 2, 2, 2, 2))
        self.riemann_tensor: np.array = np.zeros((self.points.shape[0], 2, 2, 2, 2))
        self.__compute_curvature()

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

    def __compute_metric_tensor(self, tangent_space_basis: np.array) -> tuple[np.array, np.array]:
        """
        Computes the Riemannian metric tensor and its inverse at a point from the given tangent basis vectors.

        Args:
            tangent_space_basis (np.array): A (2 x 3) array of orthonormal tangent vectors.

        Returns:
            tuple[np.array, np.array]: A (2 x 2) symmetric matrix representing the local metric tensor and 
                a (2 x 2) symmetric matrix representing the local INVERSE metric tensor.
        """
        e1, e2 = tangent_space_basis
        
        g: np.array = np.array([
            [np.dot(e1, e1), np.dot(e1, e2)],
            [np.dot(e2, e1), np.dot(e2, e2)]
        ])
        
        g_inv: np.array = np.linalg.inv(g)
        
        return g, g_inv

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
            self.metric_tensor[i], self.metric_tensor_inv[i] = self.__compute_metric_tensor(tangent_space_basis)
            
        # Normalized normal bundle
        norms: np.array = np.linalg.norm(self.normal_bundle, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.normalized_normal_bundle: np.array = self.normal_bundle / norms

    def __compute_metric_tensor_derivatives(self) -> None:
        """
        Approximates the partial derivatives of the metric tensor at each point
        using finite differences in the local tangent directions.

        For each node:
        - Iterates over its neighbors.
        - Computes directional derivatives in the basis of local tangent vectors.
        - Accumulates contributions to the derivatives of the metric tensor.

        The result is stored in `self.metric_tensor_derivatives` with shape
        (N x 2 x 2 x 2), where the second index represents derivatives in
        the two tangent directions.
        """
        for node in self.graph.nodes:
            
            g_i: np.array = self.metric_tensor[node]
            v_i: np.array = self.points[node]
            neighbors = list(self.graph.neighbors(node))
            
            sum_mu: np.array = np.zeros((2, 2))
            sum_nu: np.array = np.zeros((2, 2))
            
            for neighbor in neighbors:
                
                g_j: np.array = self.metric_tensor[neighbor]
                v_j: np.array = self.points[neighbor]
                
                g_diff: np.array = g_j - g_i
                v_alpha: np.array = v_j - v_i
                v_alpha /= np.linalg.norm(v_alpha)
                
                e1: np.array = self.tangent_bundle[node][0]
                e2: np.array = self.tangent_bundle[node][1]
                
                sum_mu += g_diff * np.dot(v_alpha, e1)
                sum_nu += g_diff * np.dot(v_alpha, e2)
            
            self.metric_tensor_derivatives[node, 0] = sum_mu
            self.metric_tensor_derivatives[node, 1] = sum_nu
    
    def __compute_christoffel_symbols(self) -> None:
        """
        Computes the Christoffel symbols of the second kind at each point
        from the metric tensor and its derivatives.

        Uses the standard formula for Christoffel symbols in coordinates:
            Γ^σ_{μν} = 1/2 * g^σl ( ∂_μ g_νl + ∂_ν g_μl - ∂_l g_iμν )
            
        Stores the resulting symbols in `self.christoffel_symbols`, with shape
        (N x 2 x 2 x 2), where:
            - N is the number of sample points.
            - The last three indices correspond to (μ, ν, σ) = Γ^σ_{μν}.
        """
        for i in range(len(self.points)):
            
            for mu in range(0, self.metric_tensor.shape[1]):
                for nu in range(0, self.metric_tensor.shape[2]):
                    for sigma in range(0, 2):
                        
                        g_inv: np.array = self.metric_tensor_inv[i]
                        partial_mu: np.array = self.metric_tensor_derivatives[i, 0]
                        partial_nu: np.array = self.metric_tensor_derivatives[i, 1]
                        
                        sum: float = 0
                        for l in range(0, 2):
                            partial_lambda: np.array = self.metric_tensor_derivatives[i][l]
                            sum += g_inv[sigma, l] * (partial_mu[nu, l] + partial_nu[mu, l] - partial_lambda[mu, nu])  
                        
                        self.christoffel_symbols[i, mu, nu, sigma] = 0.5 * sum
    
    def __compute_christoffel_symbols_derivatives(self) -> None:
        """
        Approximates the partial derivatives of the Christoffel symbols at each point
        using finite differences along the local tangent directions.

        For each node:
        - Iterates over neighboring nodes in the connectivity graph.
        - Computes directional differences of Christoffel symbols between neighbors.
        - Projects the difference onto the local tangent basis.
        - Accumulates the result as an approximation of the partial derivatives.

        The result is stored in `self.christoffel_symbols_derivatives` with shape
        (N x 2 x 2 x 2 x 2), where:
            - N is the number of sample points.
            - The first index denotes the derivative direction (tangent basis direction).
            - The next three indices correspond to (μ, ν, σ) = ∂_α Γ^σ_{μν}.
        """
        for node in self.graph.nodes:
            
            gamma_i: np.array = self.christoffel_symbols[node]
            v_i: np.array = self.points[node]
            neighbors = list(self.graph.neighbors(node))
            
            sum_mu: np.array = np.zeros((2, 2, 2))
            sum_nu: np.array = np.zeros((2, 2, 2))
            
            for neighbor in neighbors:
                
                gamma_j: np.array = self.christoffel_symbols[neighbor]
                v_j: np.array = self.points[neighbor]
                
                gamma_diff: np.array = gamma_j - gamma_i
                v_alpha: np.array = v_j - v_i
                v_alpha /= np.linalg.norm(v_alpha)
                
                e1: np.array = self.tangent_bundle[node][0]
                e2: np.array = self.tangent_bundle[node][1]
                
                sum_mu += gamma_diff * np.dot(v_alpha, e1)
                sum_nu += gamma_diff * np.dot(v_alpha, e2)
            
            self.christoffel_symbols_derivatives[node, 0] = sum_mu
            self.christoffel_symbols_derivatives[node, 1] = sum_nu
        
    def __compute_riemann_tensor(self) -> None:
        pass
    
    def __compute_curvature(self) -> None:
        """
        Computes all quantities needed to define curvature:
        - Derivatives of the metric tensor.
        - Christoffel symbols of the second kind.
        - (Optionally) derivatives of the Christoffel symbols or curvature tensors.

        Called automatically during initialization. Results are stored in:
        - `self.metric_tensor_derivatives`
        - `self.christoffel_symbols`
        - `self.christoffel_symbols_derivatives` (placeholder for future computation)
        """
        self.__compute_metric_tensor_derivatives()
        self.__compute_christoffel_symbols()
        self.__compute_christoffel_symbols_derivatives()

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