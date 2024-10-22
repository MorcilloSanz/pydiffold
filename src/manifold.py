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
    

    def compute_laplace_beltrami(self, phi: np.array) -> np.array:
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
        laplace_beltrami: list[float] = [0] * (rows * cols)

        get: float = lambda i, j: 0 if i < 0 or i >= rows or j < 0 or j >= cols else phi[i + j * cols]
        angle: float = lambda u, v: math.acos(np.dot(u, v) / np.linalg.norm(u) * np.linalg.norm(v))
        cot: float = lambda x: math.cos(x) / math.sin(x)

        for i in range(rows):
            for j in range(cols):

                sum: float = 0
                phi_ij: float = get(i, j)

                sum_cots: float = 2 * cot(angle([1,1], [1,0]))
                north_diff: float = get(i, j - 1) - phi_ij
                south_diff: float = get(i, j + 1) - phi_ij
                east_diff: float =  get(i - 1, j) - phi_ij
                west_diff: float =  get(i + 1, j) - phi_ij

                # Cotangent laplacian formula. As the mesh is uniform, all the angles
                # alpha and beta are the same.
                sum += sum_cots * (north_diff + south_diff + east_diff + west_diff)
                laplace_beltrami[i + j * cols] = sum / 2.0

        return np.array(laplace_beltrami)


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