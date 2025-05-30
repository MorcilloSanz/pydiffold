import numpy as np

from scipy.spatial import cKDTree

from .manifold import Manifold


class Function:
    """
    Base class representing a function defined on a manifold.

    Attributes:
        manifold (Manifold): The manifold on which the function is defined.
        values (np.array): Array holding function values at each point of the manifold.
    """
    
    def __init__(self, manifold: Manifold) -> None:
        """
        Initializes the function with a given manifold.
        
        Args:
            manifold (Manifold): The manifold on which the function is defined.
        """
        self.manifold = manifold
        self.values: np.array = None

    def set_value(self, value: any, index: int) -> None:
        """
        Sets the value of the function at a specific index.

        Args:
            value (any): The value to set.
            index (int): The index at which to set the value.
        """
        if self.values is not None:
            self.values[index] = value
    
    def get_value(self, index: int) -> any:
        """
        Retrieves the value of the function at a specific index.

        Args:
            index (int): The index from which to retrieve the value.

        Returns:
            any: The value at the specified index, or None if values are not initialized.
        """
        if self.values is not None:
            return self.values[index]
        return None
    

class ScalarField(Function):
    """
    Represents a scalar field defined on a manifold.

    Inherits from:
        Function
    """

    def __init__(self, manifold) -> None:
        """
        Initializes the scalar field with zeros at each point of the manifold.

        Args:
            manifold (Manifold): The manifold on which the scalar field is defined.
        """
        super().__init__(manifold)
        self.values: np.array = np.zeros(manifold.points.shape[0])
        
    def compute_gradient(self) -> np.array:
        """
        Computes the gradient of the scalar field at each point on the manifold.

        The gradient is computed using the neighboring nodes in the graph 
        associated with the manifold and a weighted difference approach.

        Returns:
            np.array: An array of gradient vectors for each point in the manifold.
        """
        gradient: np.array = np.zeros((self.manifold.points.shape[0], 3))
        
        for node in self.manifold.graph.nodes:
            
            f_i: float = self.values[node]
            v_i: np.array = self.manifold.points[node]
            neighbors = list(self.manifold.graph.neighbors(node))
            
            sum: np.array = np.zeros((3,))
            for neighbor in neighbors:
                
                if neighbor == node:
                    continue
                
                f_j: float = self.values[neighbor]
                v_j: float = self.manifold.points[neighbor]
                
                w_ij = self.manifold.graph[node][neighbor].get('weight', None)
                sum += (v_j - v_i) * (f_j - f_i) / w_ij
            
            gradient[node] = sum
        
        return gradient

    def compute_surface_gradient(self) -> np.array:
        """
        Computes the surface gradient of the scalar field on the manifold.

        This method first computes the standard 3D gradient of the scalar field in the ambient space,
        then projects it onto the tangent space of the manifold at each point,
        removing the component along the normal direction.

        Returns:
            np.array: An array of shape (N, 3) containing the surface gradient vectors at each point
                        of the manifold, where N is the number of points on the manifold.
        """
        gradient: np.array = self.compute_gradient()
        
        norms: np.array = np.linalg.norm(self.manifold.normal_bundle, axis=1, keepdims=True)
        norms[norms == 0] = 1
        
        normalized_normal_bundle: np.array = self.manifold.normal_bundle / norms
        dot_product: np.array = np.einsum("ij,ij->i", gradient, normalized_normal_bundle)
        
        return gradient - dot_product[:, np.newaxis] * normalized_normal_bundle
    
    def __estimate_global_t(self):
        """
        Estimate a global t parameter for the heat kernel Laplacian using
        the mean k-nearest neighbor distance across all points in the manifold.

        Args:
            k (int): Number of neighbors to consider (excluding self).

        Returns:
            float: Value of t = (mean_k_distance / 2)^2
        """
        tree = cKDTree(self.manifold.points)
        distances, _ = tree.query(self.manifold.points, k=self.manifold.k + 1)  # include self at index 0

        # Exclude the first column (distance to self = 0)
        mean_k_distance = np.mean(distances[:, 1:])

        # Belkin–Niyogi choice for t
        t = (mean_k_distance / 2) ** 2
        return t

    def compute_laplace_beltrami(self, t=None) -> np.array:
        """
        Compute the Laplace-Beltrami operator approximation on a manifold using the heat kernel method
        as described by Belkin and Niyogi (2005, 2008) in their work on Laplacian Eigenmaps for manifold learning.

        This method estimates the Laplace-Beltrami operator at each node of the manifold graph by
        summing weighted differences of function values at neighboring points. The weights are derived 
        from the heat kernel, using the geodesic or Euclidean distances between points, scaled by a 
        diffusion parameter t.
        
        Δf(x_i) ≈ (1/t) * Σ_j exp(-d_ij² / (4t)) * (f(x_j) - f(x_i))

        Reference:
            Belkin, M., & Niyogi, P. (2005).
            "Laplacian Eigenmaps for Dimensionality Reduction and Data Representation"
            https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf

        Args:
            t (float, optional): The heat kernel scale parameter controlling the neighborhood size.
                                    If None, it is estimated automatically based on the mean k-nearest neighbor distance.
                                    Smaller values of t emphasize local neighborhoods, but if too small,
                                    can lead to numerical instability or large operator values.

        Returns:
            np.array: A 1D numpy array of length equal to the number of manifold points, containing
                        the approximated Laplace-Beltrami operator values evaluated at each node.
        """
        if t is None:
            t = self.__estimate_global_t()
            
        laplace_beltrami: np.array = np.zeros((self.manifold.points.shape[0]))

        for node in self.manifold.graph.nodes:
            
            neighbors = list(self.manifold.graph.neighbors(node))
            
            sum: float = 0.0
            for j, neighbor in enumerate(neighbors):
                
                if neighbor != node:
                    d_ij = self.manifold.graph[node][neighbor].get('weight', None)
                    w_ij = np.exp(-(d_ij**2) / (4 * t))
                    sum += w_ij * (self.get_value(neighbor) - self.get_value(node))
                
            laplace_beltrami[node] = sum / t
                
        return laplace_beltrami
            