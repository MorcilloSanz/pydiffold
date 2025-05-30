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
    
    geodesic, arc_length = manifold.geodesic(0, 100)
    geodesic_coords: np.array = manifold.points[geodesic]
    
    print(f'Geodesic of arc length {arc_length}: {geodesic}')
    print(f'Geodesic vertex coordinates: {geodesic_coords}')

    # Point coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    gx = geodesic_coords[:, 0]
    gy = geodesic_coords[:, 1]
    gz = geodesic_coords[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='blue', s=0.25)
    ax.plot(gx, gy, gz, c='red', linewidth=3)

    ax.set_axis_off()
    ax.grid(False)

    plt.show()
