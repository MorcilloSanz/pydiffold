import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.function import ScalarField


ALPHA: float = 0.1
DELTA_T: float = 0.1


if __name__ == "__main__":
    
    # Load points
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/assets/surface.txt')

    # Compute manifold
    manifold: Manifold = Manifold(points)

    phi: ScalarField = ScalarField(manifold)
    for i in range(points.shape[0]):
        phi.set_value(np.sin(i), i)
        
    laplace_beltrami: np.array = phi.compute_laplace_beltrami(t=1)

    # Point coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=phi.values, cmap='plasma', s=2)
    
    def update(frame):
        phi.values = phi.values + DELTA_T * ALPHA * phi.compute_laplace_beltrami(t=1)
        sc.set_array(phi.values)
        return sc,

    # Agregar barra de color (opcional)
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)

    ax.set_axis_off()
    ax.grid(False)
    
    anim = FuncAnimation(fig, update, frames=100, interval=100, blit=False)

    plt.show()
