import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.function import ScalarField


ALPHA: float = 0.2
DELTA_T: float = 0.5

HEAT_SCALE_LAPLACIAN: float = 2


if __name__ == "__main__":
    
    # Load points
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/assets/bunny.txt')
    
    # Transform coords
    transform: np.array = np.array([
        [1, 0, 0],
        [0, 0, 1], 
        [0, 1, 0]
    ])
    
    theta_z: float = np.pi
    
    rotation_z: np.array = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0,              0,                 1]
    ])
    
    points = points @ transform.T @ rotation_z.T

    # Compute manifold
    manifold: Manifold = Manifold(points)

    # Compute phi function
    phi: ScalarField = ScalarField(manifold)
    for i in range(points.shape[0]):
        coords: np.array = manifold.points[i]
        phi.set_value(np.sin(coords[0] * 3) + np.sin(coords[1] * 3), i)

    # Point coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Plot
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=phi.values, cmap='hot', s=15)
    
    def update(frame):
        phi.values = phi.values + DELTA_T * ALPHA * phi.compute_laplace_beltrami(t=HEAT_SCALE_LAPLACIAN)
        sc.set_array(phi.values)
        return sc,

    # Agregar barra de color (opcional)
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)

    ax.set_axis_off()
    ax.grid(False)
    
    anim = FuncAnimation(fig, update, frames=100, interval=0, blit=False)

    plt.show()
