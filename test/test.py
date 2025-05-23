import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.function import ScalarField


if __name__ == "__main__":
    
    points: np.array = np.loadtxt('bunny.txt')
    manifold = Manifold(points)

    geodesic, arc_length = manifold.geodesic(0, 11016)
    print(f'Geodesic of arc length {arc_length}: {geodesic}')
    print(f'Geodesic vertex coordinates: {manifold.points[geodesic]}')
    print(f'Manifold normal bundle {manifold.normal_bundle}')
    print(f'Manifold tangent bundle {manifold.tangent_bundle}')
    print(f'Metric tensor {manifold.metric_tensor}')
    
    function: ScalarField = ScalarField(manifold)
    print(f'Function values shape {function.values.shape}')
    
    function.set_value(5.5, 10)
    print(function.get_value(10))
    
    surface_gradient: np.array = function.compute_surface_gradient()
    print(surface_gradient)