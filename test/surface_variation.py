import sys
from pathlib import Path

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.function import ScalarField


def get_colors(values: ScalarField) -> np.array:
    """
    Generate RGB colors from a scalar field using the gist_heat colormap.

    Args:
        values (ScalarField): Scalar field defined on the manifold.

    Returns:
        np.array: An array of RGB colors (Nx3), where N is the number of points.
    """
    values_min, values_max = values.min(), values.max()
    values_normalized = (values - values_min) / (values_max - values_min)

    cmap = plt.get_cmap("viridis")
    colors_rgba = cmap(values_normalized)
    colors = colors_rgba[:, :3]
    
    return colors

if __name__ == "__main__":
    
    # Load points
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/assets/bunny.txt')
    
    # Compute manifold
    manifold: Manifold = Manifold(points)

    # Create pcd for Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    colors: np.array = get_colors(manifold.surface_variation)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 3D viewer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=300, height=300)
    vis.add_geometry(pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 8
    
    # Show
    vis.run()
    vis.destroy_window()