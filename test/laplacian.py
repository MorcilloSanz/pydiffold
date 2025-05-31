import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.function import ScalarField


if __name__ == "__main__":
    
    # Load points
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/assets/surface.txt')

    # Compute manifold
    manifold: Manifold = Manifold(points)

    function: ScalarField = ScalarField(manifold)
    for i in range(points.shape[0]):
        coords: np.array = manifold.points[i]
        function.set_value(2 * (np.sin(coords[0] * 5) + np.sin(coords[1] * 5)), i)
        
    laplace_beltrami: np.array = function.compute_laplace_beltrami(t=2)

    # Point coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=laplace_beltrami, cmap='inferno', s=2)

    # Agregar barra de color (opcional)
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)

    ax.set_axis_off()
    ax.grid(False)

    plt.show()
