import math

import numpy as np
import matplotlib.pyplot as plt

from manifold import *


XY_LIMITS: tuple[int, int] = (-5, 5)
XY_POINTS: int = 25


def f(x_mesh, y_mesh):
    """
    Function to generate the Z coordinates of a 3D manifold from the X and Y mesh grids.

    This function computes the Z values based on the X and Y mesh grids using the equation:
    z = -(x^2 + x + y^2 + y). It is used to create the third dimension of the manifold.

    Parameters:
    -----------
    x_mesh : np.array
        2D array representing the X values on the mesh grid.
    y_mesh : np.array
        2D array representing the Y values on the mesh grid.

    Returns:
    --------
    np.array
        2D array representing the Z values computed from the given equation.
    """
    return -(x_mesh**2 + x_mesh + y_mesh**2 + y_mesh)


def compute_phi(manifold: Manifold) -> np.array:
    """
    Computes the scalar field `phi` on the manifold.

    This function generates a 1D array `phi`, where each value is computed using the sine of
    the row index `i` and the cosine of the column index `j`.
    phi(x,y) = sin(x) + cos(y)

    Parameters:
    -----------
    manifold : Manifold
        An instance of the Manifold class.

    Returns:
    --------
    np.array
        A 1D array representing the scalar field `phi` on the manifold.
    """
    rows, cols = manifold.x_mesh.shape
    phi: list[float] = [0] * (rows * cols)

    for i in range(rows):
        for j in range(cols):
            phi[i + j * cols] = math.sin(i) + math.cos(j) 

    return np.array(phi)


if __name__ == "__main__":

    manifold: Manifold = Manifold.create_manifold(f, XY_LIMITS, XY_LIMITS, XY_POINTS)

    phi = compute_phi(manifold)
    laplace_beltrami = manifold.laplace_beltrami(phi)

    fig = plt.figure(figsize=(12, 6))

    # Phi plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, color='blue', edgecolor='k', alpha=0.5, label='M')

    scatter1 = ax1.scatter(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, c=phi, cmap='coolwarm', s=25, label='phi')
    fig.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.7, aspect=20)

    ax1.legend(loc='upper right')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Phi')

    # Laplacian plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, color='blue', edgecolor='k', alpha=0.5, label='M')

    scatter2 = ax2.scatter(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, c=laplace_beltrami, cmap='magma', s=25, label='Laplace-Beltrami')
    fig.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.7, aspect=20)

    ax2.legend(loc='upper right')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Laplace-Beltrami')

    plt.tight_layout()
    plt.show()

