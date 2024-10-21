import math

import numpy as np
import matplotlib.pyplot as plt

from manifold import *


XY_LIMITS: tuple[int, int] = (-5, 5)
XY_POINTS: int = 15


def compute_manifold() -> Manifold:
    """
    Computes a manifold (mesh) in R^3.
    """
    x_points = np.linspace(XY_LIMITS[0], XY_LIMITS[1], XY_POINTS)
    y_points = np.linspace(XY_LIMITS[0], XY_LIMITS[1], XY_POINTS)

    x_mesh, y_mesh = np.meshgrid(x_points, y_points)
    z_mesh = -(x_mesh**2 + x_mesh + y_mesh**2 + y_mesh)

    return Manifold(x_mesh, y_mesh, z_mesh)


def compute_phi(manifold: Manifold) -> np.array:
    """
    Computes a scalarfield over the manifold M.
    phi : M -> R
    
    Args:
        manifold: the manifold M.

    Returns:
        The function phi.
    """
    rows, cols = manifold.x_mesh.shape
    phi: list[float] = [0] * (rows * cols)

    for i in range(rows):
        for j in range(cols):
            # phi function in the vertex (i, j) of the surface
            phi[i + j * cols] = math.sin(i) + math.cos(j) 

    return np.array(phi)


def compute_laplace_beltrami(phi: np.array, manifold: Manifold) -> np.array:
    """
    Computes the Laplace-Beltrami operator of the function phi over the manifold M
    using the cotangent laplacian formula.
    (uniform mesh)

    Args:
        phi: the function phi.
        manifold: the manifold M where the function phi is defined.

    Returns:
        The Laplace-Beltrami of the function phi.
    """
    rows, cols = manifold.x_mesh.shape
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


if __name__ == "__main__":

    manifold: Manifold = compute_manifold()
    phi = compute_phi(manifold)
    laplace_beltrami = compute_laplace_beltrami(phi, manifold)

    fig = plt.figure(figsize=(12, 6))

    # Phi plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, color='blue', edgecolor='k', alpha=0.5, label='M')

    scatter1 = ax1.scatter(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, c=phi, cmap='coolwarm', s=25, label='phi(x,y,z)')
    fig.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.7, aspect=20)

    ax1.legend(loc='upper right')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Phi')

    # Laplacian plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, color='blue', edgecolor='k', alpha=0.5, label='M')

    scatter2 = ax2.scatter(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, c=laplace_beltrami, cmap='magma', s=25, label='Laplace Beltrami')
    fig.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.7, aspect=20)

    ax2.legend(loc='upper right')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Laplace-Beltrami')

    plt.tight_layout()
    plt.show()

