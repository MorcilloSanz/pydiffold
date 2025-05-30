import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold


if __name__ == "__main__":
    
    # Load points
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/bunny.txt')

    # Transform coords
    transform: np.array = np.array([
        [1, 0, 0],
        [0, 0, 1], 
        [0, 1, 0]
    ])
    
    points = points @ transform.T
    points = points[:5000] # subsample

    # Compute manifold
    manifold: Manifold = Manifold(points)

    # Point coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Normal components
    u = manifold.normal_bundle[:, 0]
    v = manifold.normal_bundle[:, 1]
    w = manifold.normal_bundle[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='blue', s=0.25)
    ax.quiver(x, y, z, u, v, w, length=0.01, normalize=True, color='red', linewidth=0.3)

    ax.set_axis_off()
    ax.grid(False)

    plt.show()
