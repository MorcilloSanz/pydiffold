import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.function import ScalarField


if __name__ == "__main__":
    
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/bunny.txt')
    manifold = Manifold(points)

    geodesic, arc_length = manifold.geodesic(0, 11016)
    print(f'\033[1;95mGeodesic of arc length {arc_length}:\033[0m\n {geodesic}')
    print(f'\033[1;95mGeodesic vertex coordinates:\033[0m\n {manifold.points[geodesic]}')
    print(f'\033[1;95mManifold normal bundle:\033[0m\n {manifold.normal_bundle}')
    print(f'\033[1;95mManifold tangent bundle:\033[0m\n {manifold.tangent_bundle}')
    print(f'\033[1;95mMetric tensor:\033[0m\n {manifold.metric_tensor}')
    
    function: ScalarField = ScalarField(manifold)
    print(f'\033[1;95mFunction values shape:\033[0m\n {function.values.shape}')
    
    function.set_value(5.5, 10)
    print(function.get_value(10))
    
    surface_gradient: np.array = function.compute_surface_gradient()
    print(surface_gradient)