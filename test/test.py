import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.function import ScalarField


if __name__ == "__main__":
    
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/assets/bunny.txt')
    manifold = Manifold(points)

    geodesic, arc_length = manifold.geodesic(0, 2000)
    print(f'\033[1;95mGeodesic of arc length {arc_length}:\033[0m\n {geodesic}')
    print(f'\033[1;95mGeodesic vertex coordinates:\033[0m\n {manifold.points[geodesic]}')
    print(f'\033[1;95mManifold normal bundle:\033[0m\n {manifold.normal_bundle}')
    print(f'\033[1;95mManifold tangent bundle:\033[0m\n {manifold.tangent_bundle}')
    print(f'\033[1;95mMetric tensor:\033[0m\n {manifold.metric_tensor}')
    
    function: ScalarField = ScalarField(manifold)
    for i in range(points.shape[0]):
        function.set_value(np.random.uniform(0, 10), i)
        
    print(f'\033[1;95mFunction values shape:\033[0m\n {function.values.shape}')
    
    surface_gradient: np.array = function.compute_surface_gradient()
    print(f'\033[1;95mSurface gradient:\033[0m\n {surface_gradient}')
    
    laplace_beltrami: np.array = function.compute_laplace_beltrami(t=1)
    print(f'\033[1;95mLaplace Beltrami:\033[0m\n {laplace_beltrami}')