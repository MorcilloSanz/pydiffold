# PyDiffold :earth_americas:
`PyDiffold` is a Python library for `differential geometry`. It provides tools for approximating `local differential structure` (tangent spaces, normal vectors, Riemannian metric tensors...), as well as discrete `differential operators` such as surface gradients and the Laplace–Beltrami operator, using only point cloud data.

The library is designed to facilitate numerical experimentation in geometry processing and geometric PDEs by operating directly on sampled surfaces, 
without requiring explicit mesh connectivity.

## Heat and Wave Equation on the Stanford Bunny
The Laplace–Beltrami operator was computed on the vertices of the Stanford Bunny, without relying on mesh connectivity. Based on this operator, the heat equation and the wave equation were solved on the surface of the model.

The resulting solutions are shown as GIF animations, illustrating the diffusion of heat and the propagation of waves over the geometry.

<p align="center">
  <img src="/img/heat_equation_1.gif" width="200"/>
  <img src="/img/heat_equation_2.gif" width="200"/>
  <img src="/img/wave_equation_1.gif" width="200"/>
  <img src="/img/wave_equation_2.gif" width="200"/>
</p>

**Code Snippets**
```python
# Load points
test_path: str = str(Path(__file__).resolve().parent)
points: np.array = np.loadtxt(test_path + '/assets/bunny.txt')

# Compute manifold
manifold: Manifold = Manifold(points)

# Geodesic
geodesic, arc_length = manifold.geodesic(0, 2000)
geodesic_coords: np.array = manifold.points[geodesic]
```

```python
# Compute phi function
phi: ScalarField = ScalarField(manifold)
for i in range(points.shape[0]):
    coords: np.array = manifold.points[i]
    phi.set_value(np.sin(coords[0] * 2), i)

# Laplace-Beltrami
laplacian: np.array = phi.compute_laplace_beltrami(t=HEAT_SCALE_LAPLACIAN)
```

## Features
* **Manifold graph:** computes a graph $G = (N,E)$ with associating point indices and distances.
* **Compute normals:** estimates normal vectors using PCA.
* **Compute tangent bundle:** computes the tangent bundle $TM$ using PCA.
* **Compute metric tensor:** computes the metric tensor $g_{\mu \nu}$ for each point $p$ of the manifold.
* **Compute geodesics:** computes the shortest path $\gamma(t)$ between two points of the manifold and its arc length $L$.
* **Define scalar fields in manifolds:** $\phi : \mathcal{M} \rightarrow \mathbb{R}$.
* **Compute gradient:** approximates the gradient $\nabla f$ of a scalar field defined in a manifold.
* **Compute surface gradient:** computes the surface gradient $\nabla_M f$ of a scalar field defined in a manifold.
* **Compute Laplace-Beltrami:** approximates the Laplace-Beltrami $\Delta f$ of a scalar field defined in a manifold.

## TODO
* Vector and Tensor fields
* Covariant Derivative
* Christoffel symbols
* Riemann Curvature Tensor
* Ricci tensor
* Higher dimensions manifolds

## Dependencies
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* [NetworkX](https://github.com/networkx/networkx)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [Open3D](https://github.com/isl-org/Open3D)