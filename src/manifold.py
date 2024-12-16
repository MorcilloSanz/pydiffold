"""
MIT License

Copyright (c) 2024 Alberto Morcillo Sanz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/MorcilloSanz/Laplace-Beltrami
"""

import math
import numpy as np


class Manifold:
    """
    Class representing a 3D manifold created from a mathematical function.
    """

    def __init__(self, x_mesh: np.array, y_mesh: np.array, z_mesh: np.array) -> None:
        """
        Constructor for the Manifold class.

        Parameters:
        -----------
        x_mesh : np.array
            2D array containing the values of the mesh along the X axis.
        y_mesh : np.array
            2D array containing the values of the mesh along the Y axis.
        z_mesh : np.array
            2D array containing the values of the mesh along the Z axis (computed from a function).
        """
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.z_mesh = z_mesh


    @staticmethod
    def create_manifold(f, x_limits: tuple[float, float], y_limits: tuple[float, float], xy_points: int):
        """
        Static method to create a manifold from a given function and boundary limits.

        Parameters:
        -----------
        f : callable
            A function of two variables (x, y) that returns the corresponding z value.
        x_limits : tuple[float, float]
            A tuple representing the minimum and maximum values for the x-axis.
        y_limits : tuple[float, float]
            A tuple representing the minimum and maximum values for the y-axis.
        xy_points : int
            The number of points to sample in both the x and y directions.

        Returns:
        --------
        Manifold
            A new instance of the Manifold class with the computed mesh grids.
        """
        x_points = np.linspace(x_limits[0], x_limits[1], xy_points)
        y_points = np.linspace(y_limits[0], y_limits[1], xy_points)

        x_mesh, y_mesh = np.meshgrid(x_points, y_points)
        z_mesh = f(x_mesh, y_mesh)

        return Manifold(x_mesh, y_mesh, z_mesh)
    

    def coords(self, i: int, j: int) -> np.array:
        """
        Retrieve the coordinates of a point on the manifold at a given index.

        Parameters:
        -----------
        i : int
            The row index for the point in the mesh grid.
        j : int
            The column index for the point in the mesh grid.

        Returns:
        --------
        np.array
            A 1D array containing the [x, y, z] coordinates of the point
            located at position (i, j) in the manifold's mesh grid.
        """
        return np.array([self.x_mesh[i, j], self.y_mesh[i, j], self.z_mesh[i, j]])


    def surface_gradient(self, phi: np.array) -> np.array:
        """
        Compute the surface gradient of a scalar field defined on a 3D manifold.

        This function calculates the gradient of the scalar field `phi` on the surface of the
        manifold by considering the differences between adjacent points and using the tangent vectors
        to represent the surface geometry in three-dimensional space.

        Parameters:
        -----------
        phi : np.array
            A 1D array representing the scalar field values at each point in the 3D mesh grid.
            The length of `phi` should be equal to the total number of points in the mesh grid
            (rows * cols).

        Returns:
        --------
        np.array
            A 2D array of shape (rows * cols, 3) where each row corresponds to the surface gradient
            vector at the respective point in the mesh grid. Each gradient vector is represented
            in 3D space as [gx, gy, gz].

        Notes:
        ------
        - The function does not compute the gradient at the boundary points (i.e., when
        `i` or `j` is at the edge of the mesh grid). For these points, the gradient is left
        as zero.
        - The gradients are computed by taking the differences between the values of `phi` at 
        adjacent points and scaling these differences by the normalized tangent vectors that
        correspond to the directions of the differences.
        - The tangent vectors are computed by obtaining the coordinates of adjacent points
        using the `coords` method, ensuring that the gradient reflects the surface's geometry.

        Example:
        ---------
        >>> phi = np.array([...])  # Some scalar field values in 3D space
        >>> gradient = manifold.surface_gradient(phi)
        >>> print(gradient)  # Outputs the surface gradient at each point in the manifold.
        """
        rows, cols = self.x_mesh.shape
        gradient: np.array = np.zeros((rows * cols, 3))

        get: float = lambda i, j: 0 if i < 0 or i >= rows or j < 0 or j >= cols else phi[i + j * cols]

        for i in range(rows):
            for j in range(cols):

                # Boundary conditions
                if i <= 0 or j <= 0 or i >= rows - 1 or j >= cols - 1:
                    continue

                neighbors: list[tuple[int, int]] = [
                    (i - 1, j - 1), (i   , j - 1), (i + 1, j - 1),
                    (i - 1,     j),                (i + 1,     j),
                    (i - 1, j + 1), (i   , j + 1), (i + 1, j + 1)
                ]

                phi_ij: float = get(i, j)
                coords_ij: np.array = self.coords(i, j)

                grad: np.array = np.zeros(3)
                for neighbor in neighbors:

                    # Compute difference
                    diff: float = get(neighbor[0], neighbor[1]) - phi_ij

                    # Compute tangent vector
                    tangent_vector: np.array = self.coords(neighbor[0], neighbor[1]) - coords_ij
                    tangent_vector /= np.linalg.norm(tangent_vector)

                    # Project the derivative onto the vector
                    grad += diff * tangent_vector

                # Compute gradient
                gradient[i + j * cols] = grad

        return gradient


    def laplace_beltrami(self, phi: np.array) -> np.array:
        """
        Computes the Laplace-Beltrami operator on the manifold for a given scalar field `phi`.

        The Laplace-Beltrami operator is a generalization of the Laplacian to curved surfaces
        and is used to describe the diffusion of scalar fields on the manifold.

        Parameters:
        -----------
        phi : np.array
            A 1D array representing the scalar field on the manifold (same size as the mesh).

        Returns:
        --------
        np.array
            A 1D array representing the Laplace-Beltrami operator applied to the scalar field `phi`.
        """
        rows, cols = self.x_mesh.shape
        laplace_beltrami = np.zeros_like(phi)

        phi_reshaped = phi.reshape((rows, cols))

        # Precompute cotangent weights (constant for uniform mesh)
        uniform_cot = 2 * (1 / np.sqrt(3))  # cot(pi/3) = 1/sqrt(3)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):

                north_diff = phi_reshaped[i, j - 1] - phi_reshaped[i, j]
                south_diff = phi_reshaped[i, j + 1] - phi_reshaped[i, j]
                west_diff = phi_reshaped[i - 1, j] - phi_reshaped[i, j]
                east_diff = phi_reshaped[i + 1, j] - phi_reshaped[i, j]

                laplace_beltrami[i * cols + j] = uniform_cot * (
                    north_diff + south_diff + west_diff + east_diff
                )

        return laplace_beltrami


    def non_uniform_laplace_beltrami(self, phi: np.array) -> np.array:
        """
        Computes the Laplace-Beltrami operator on a non-uniform manifold for a given scalar field `phi`.

        This method generalizes the Laplace-Beltrami operator to non-uniform meshes by accounting for 
        varying angles between the mesh points, using the cotangent formula for each point on the manifold.

        Parameters:
        -----------
        phi : np.array
            A 1D array representing the scalar field on the manifold (same size as the mesh grid).

        Returns:
        --------
        np.array
            A 1D array representing the Laplace-Beltrami operator applied to the scalar field `phi`, 
            accounting for the non-uniform geometry of the mesh.
        """
        rows, cols = self.x_mesh.shape
        laplace_beltrami = np.zeros_like(phi)
        phi_reshaped = phi.reshape((rows, cols))

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):

                coords_center = self.coords(i, j)

                def cotangent_weights(i1, j1, i2, j2):
                    u = self.coords(i1, j1) - coords_center
                    v = self.coords(i2, j2) - coords_center
                    angle = np.arccos(
                        np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
                    )
                    return 1 / np.tan(angle)

                north_cots = cotangent_weights(i - 1, j - 1, i, j - 1) + cotangent_weights(i + 1, j - 1, i, j - 1)
                south_cots = cotangent_weights(i - 1, j + 1, i, j + 1) + cotangent_weights(i + 1, j + 1, i, j + 1)
                west_cots = cotangent_weights(i - 1, j - 1, i - 1, j) + cotangent_weights(i - 1, j + 1, i - 1, j)
                east_cots = cotangent_weights(i + 1, j - 1, i + 1, j) + cotangent_weights(i + 1, j + 1, i + 1, j)

                north_diff = phi_reshaped[i, j - 1] - phi_reshaped[i, j]
                south_diff = phi_reshaped[i, j + 1] - phi_reshaped[i, j]
                west_diff = phi_reshaped[i - 1, j] - phi_reshaped[i, j]
                east_diff = phi_reshaped[i + 1, j] - phi_reshaped[i, j]

                width = np.linalg.norm(self.coords(i + 1, j) - self.coords(i - 1, j))
                height = np.linalg.norm(self.coords(i, j + 1) - self.coords(i, j - 1))
                area = abs(width * height)

                laplace_beltrami[i * cols + j] = (
                    north_cots * north_diff +
                    south_cots * south_diff +
                    west_cots * west_diff +
                    east_cots * east_diff
                ) / (2.0 * area)

        return laplace_beltrami


    @property
    def x(self) -> np.array:
        """
        Returns the flattened 1D array of x values from the mesh grid.

        Returns:
        --------
        np.array
            1D array of x values.
        """
        return self.x_mesh.flatten()
    

    @property
    def y(self) -> np.array:
        """
        Returns the flattened 1D array of y values from the mesh grid.

        Returns:
        --------
        np.array
            1D array of y values.
        """
        return self.y_mesh.flatten()
    
    
    @property
    def z(self) -> np.array:
        """
        Returns the flattened 1D array of z values computed from the function.

        Returns:
        --------
        np.array
            1D array of z values.
        """
        return self.z_mesh.flatten()