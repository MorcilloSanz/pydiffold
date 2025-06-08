import sys
from pathlib import Path

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.function import ScalarField


C: float = 0.5
DELTA_T: float = 0.5
HEAT_SCALE_LAPLACIAN: float = 2

animation_running: bool = False


def get_colors(phi: ScalarField) -> np.array:
    """
    Generate RGB colors from a scalar field using the hsv colormap.

    Args:
        phi (ScalarField): Scalar field defined on the manifold.

    Returns:
        np.array: An array of RGB colors (Nx3), where N is the number of points.
    """
    # Normalize phi values to [0,1] for colormap
    phi_values = phi.values
    phi_min, phi_max = phi_values.min(), phi_values.max()
    phi_normalized = (phi_values - phi_min) / (phi_max - phi_min)

    # Generate colors using matplotlib's colormap
    cmap = plt.get_cmap("jet")
    colors_rgba = cmap(phi_normalized)  # Returns RGBA colors
    colors = colors_rgba[:, :3]  # Use only RGB
    
    return colors


def solve_equation(phi: ScalarField, phi_minus_1: ScalarField, pcd: o3d.geometry.PointCloud, move_points=False) -> None:
    """
    Perform one step of the wabe equation and update the point cloud colors.

    Args:
        phi (ScalarField): The scalar field to be updated via the heat equation.
        phi_minus_1 (ScalarField): The scalar field at t=-1
        pcd (o3d.geometry.PointCloud): The point cloud whose colors are updated.
    """
    # Solve equation
    laplacian: np.aray = phi.compute_laplace_beltrami(t=HEAT_SCALE_LAPLACIAN)
    phi_t1: np.array = 2 * phi.values - phi_minus_1.values + (C * DELTA_T)**2 * laplacian
    
    phi_minus_1.values = np.copy(phi.values)
    phi.values = phi_t1

    # Update colors
    pcd.colors = o3d.utility.Vector3dVector(get_colors(phi))
    
    # Move points according to the solutions
    if move_points:
        
        norms: np.array = np.linalg.norm(phi.manifold.normal_bundle, axis=1, keepdims=True)
        norms[norms == 0] = 1
        
        normalized_normal_bundle: np.array = phi.manifold.normal_bundle / norms
        points: np.array = phi.manifold.points + normalized_normal_bundle * phi.values[:, np.newaxis]
        pcd.points = o3d.utility.Vector3dVector(points)


def toggle_animation(vis):
    """
    Key callback to start the animation when 'A' is pressed.
    """
    global animation_running
    animation_running = not animation_running
    return False  # Return False to keep the window open


if __name__ == "__main__":
    
    print('\033[1;95mToggle animation pressing A\033[0m')
    
    # Load points
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/assets/bunny.txt')
    
    # Compute manifold
    manifold: Manifold = Manifold(points)

    # Compute phi function
    phi: ScalarField = ScalarField(manifold)
    for i in range(points.shape[0]):
        coords: np.array = manifold.points[i]
        phi.set_value((np.sin(coords[0]) + np.sin(coords[1])) / 3, i)
        
    phi_minus_1: ScalarField = ScalarField(manifold)
    phi_minus_1.values = np.copy(phi.values)

    # Create pcd for Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    colors: np.array = get_colors(phi)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 3D viewer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=800, height=800)
    vis.add_geometry(pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 3
    
    # Animation callback
    def timer_callback(vis):
        if animation_running:
            solve_equation(phi, phi_minus_1, pcd, move_points=True)
            vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        return False
    
    vis.register_animation_callback(timer_callback)
    vis.register_key_callback(ord("A"), toggle_animation)

    # Show
    vis.run()
    vis.destroy_window()