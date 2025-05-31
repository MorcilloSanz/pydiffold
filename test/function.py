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

    # Point coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Plot
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=function.values, cmap='jet', s=1)

    # Agregar barra de color (opcional)
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)

    ax.set_axis_off()
    ax.grid(False)

    plt.show()
