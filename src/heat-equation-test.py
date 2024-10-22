import math

import numpy as np
import matplotlib.pyplot as plt

from manifold import *


XY_LIMITS: tuple[int, int] = (-5, 5)
XY_POINTS: int = 15


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


def solve_heat_equation(phi: np.array, manifold: Manifold, alpha: float = 1.0, delta_t: float = 1.0) -> None:
    """
    Solves the heat equation on a given manifold over one time step.

    This function updates the scalar field `phi` based on the heat equation, using the 
    Laplace-Beltrami operator to compute the diffusion of heat over the surface of the manifold.

    The heat equation is given by:
        ∂phi/∂t = alpha * Δ(phi)
    where `alpha` is the thermal diffusivity, and Δ(phi) is the Laplace-Beltrami operator applied to `phi`.

    Parameters:
    -----------
    phi : np.array
        A 1D array representing the initial scalar field (temperature distribution) on the manifold.
    manifold : Manifold
        An instance of the `Manifold` class, containing the mesh grids and a method to compute the Laplace-Beltrami operator.
    alpha : float, optional (default=1.0)
        The thermal diffusivity constant, controlling the rate of heat diffusion.
    delta_t : float, optional (default=1.0)
        The time step for the heat equation update.

    Returns:
    --------
    np.array
        A 1D array representing the updated scalar field `phi` after one time step.
    """
    laplace_beltrami = manifold.compute_laplace_beltrami(phi)
    rows, cols = manifold.x_mesh.shape

    for i in range(rows):
        for j in range(cols):

            index: int = i + j * cols

            phi_ij = phi[index]
            laplacian_ij = laplace_beltrami[index]

            phi[index] = phi_ij + alpha * delta_t * laplacian_ij

    return np.array(phi)


if __name__ == "__main__":
    
    manifold: Manifold = Manifold.create_manifold(f, XY_LIMITS, XY_LIMITS, XY_POINTS)
    phi = compute_phi(manifold)

    conductivity: float = 1.0
    delta_t: float = 0.5

    # Create figure
    fig = plt.figure(figsize=(8, 8))

    # Phi plot at t=0
    ax1 = fig.add_subplot(221, projection='3d')  # 2x2 grid, first plot
    ax1.plot_surface(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, color='blue', edgecolor='k', alpha=0.5, label='M')

    scatter1 = ax1.scatter(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, c=phi, cmap='inferno', s=25, label='phi(x,y,z)')
    fig.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.7, aspect=20)

    ax1.legend(loc='upper right')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Phi at t=0')

    # Solve heat equation t=0.5
    solve_heat_equation(phi, manifold, conductivity, delta_t)

    # Phi plot at t=0.5
    ax2 = fig.add_subplot(222, projection='3d')  # 2x2 grid, second plot
    ax2.plot_surface(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, color='blue', edgecolor='k', alpha=0.5, label='M')

    scatter2 = ax2.scatter(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, c=phi, cmap='inferno', s=25, label='phi(x,y,z)')
    fig.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.7, aspect=20)

    ax2.legend(loc='upper right')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Phi at t=0.5')

    # Solve heat equation t=1
    solve_heat_equation(phi, manifold, conductivity, delta_t)

    # Phi plot at t=1
    ax3 = fig.add_subplot(223, projection='3d')  # 2x2 grid, third plot
    ax3.plot_surface(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, color='blue', edgecolor='k', alpha=0.5, label='M')

    scatter3 = ax3.scatter(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, c=phi, cmap='inferno', s=25, label='phi(x,y,z)')
    fig.colorbar(scatter3, ax=ax3, pad=0.1, shrink=0.7, aspect=20)

    ax3.legend(loc='upper right')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Phi at t=1')

    # Solve heat equation t=1.5
    solve_heat_equation(phi, manifold, conductivity, delta_t)

    # Phi plot at t=1.5
    ax4 = fig.add_subplot(224, projection='3d')  # 2x2 grid, fourth plot
    ax4.plot_surface(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, color='blue', edgecolor='k', alpha=0.5, label='M')

    scatter4 = ax4.scatter(manifold.x_mesh, manifold.y_mesh, manifold.z_mesh, c=phi, cmap='inferno', s=25, label='phi(x,y,z)')
    fig.colorbar(scatter4, ax=ax4, pad=0.1, shrink=0.7, aspect=20)

    ax4.legend(loc='upper right')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Phi at t=1.5')

    # Show
    plt.tight_layout()
    plt.show()
