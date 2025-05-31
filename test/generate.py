from pathlib import Path

import numpy as np


if __name__ == "__main__":
    
    assets_path: str = str(Path(__file__).resolve().parent) + '/assets/'
    
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x, y)

    # Surface z = f(x,y)
    r2 = x**2 + y**2
    z = (1 - r2) * np.exp(-0.5 * r2)

    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    np.savetxt(assets_path + "surface.txt", points, fmt="%.6f")