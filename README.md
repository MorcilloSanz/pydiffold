# PyDiffold :earth_americas:
`PyDiffold` is a Python library for `differential geometry`. It provides tools for approximating `local differential structure`, as well as discrete `differential operators`.

The library is designed to facilitate numerical experimentation in geometry processing and geometric PDEs by operating directly on sampled surfaces, without requiring explicit mesh connectivity.

### PyDiffold for Partial Differential Equations
The `Laplace–Beltrami operator` was computed on the vertices of the Stanford Bunny, without relying on mesh connectivity. Based on this operator, the heat equation and the wave equation were solved on the surface of the model.

The resulting solutions are shown as GIF animations, illustrating the diffusion of heat and the propagation of waves over the geometry.

<p align="center">
  <img src="/img/heat_equation_1.gif" width="200"/>
  <img src="/img/heat_equation_2.gif" width="200"/>
  <img src="/img/wave_equation_1.gif" width="200"/>
  <img src="/img/wave_equation_2.gif" width="200"/>
</p>

## Local differential structure
The `Manifold` class is the core of the library, representing a 2D surface discretized by a set of sample points. It automatically computes various fundamental geometric quantities:
* **Conectivity Graph:** Builds a graph representing the neighborhood relationships between sample points, crucial for discrete approximations of differential operators.
* **Tangent and normal bundles:** Estimates the local tangent plane and surface normal at each point using Principal Component Analysis (PCA) on neighboring points.
* **Riemannian Metric tensor:** Calculates the metric tensor and its inverse at each point, which defines distances and angles on the manifold.
* **Christoffel symbols (second kind):** Computes these symbols, which describe how the basis vectors change across the manifold, essential for defining covariant derivatives.
* **Riemann curvature tensor:** Quantifies the intrinsic curvature of the manifold, indicating how much a manifold deviates from being flat.
* **Ricci Curvature Tensor and Ricci Scalar:** Derived from the Riemann tensor, these provide information about the average curvature in different directions.
* **Gaussian Curvature:** For 2D manifolds, this scalar value fully characterizes the intrinsic curvature at each point.
* **Compute geodesics:** Computes the shortest path between two points of the manifold and its arc length.

### Code snippets
To use this library, you'll need to provide a set of 3D points representing your manifold. The library will then automatically compute its geometric properties.
```python
points: np.array = np.loadtxt('/assets/bunny.txt')                       # (N, 3)
m: Manifold = Manifold(points)

normal_bundle: np.array = m.normal_bundle                                # (N, 3)
tangent_bundle: np.array = m.tangent_bundle                              # (N, 2, 3)

metric_tensor: np.array = m.metric_tensor                                # (N, 2, 2)
metric_tensor_inv: np.array = m.metric_tensor_inv                        # (N, 2, 2)
metric_tensor_derivatives: np.array = m.metric_tensor_derivatives        # (N, 2, 2, 2)

chris_symbols: np.array = m.christoffel_symbols                          # (N, 2, 2, 2)
chris_symbols_deivatives: np.array = m.christoffel_symbols_derivatives   # (N, 2, 2, 2, 2)

riemann_tensor: np.array = m.riemann_tensor                              # (N, 2, 2, 2, 2)
ricci_tensor: np.array = m.ricci_tensor                                  # (N, 2, 2)
ricci_scalar: np.array = m.ricci_scalar                                  # (N,)
gauss_curvature: np.array = m.gauss_curvature                            # (N,)
```
```python
geodesic, arc_length = m.geodesic(0, 2000)                               # (K,)
geodesic_coords: np.array = m.points[geodesic]                           # (K, 3)
```

## Differential operators
The `ScalarField` class allows you to define and operate on scalar-valued functions over the manifold:
* **Gradient:** Computes the standard 3D gradient of the scalar field in the ambient space.
* **Surface Gradient** Projects the ambient-space gradient onto the manifold's tangent space, yielding the true gradient on the surface.
* **Partial and Directional Derivatives:** Calculates derivatives of the scalar field along the local tangent basis vectors.
* **Laplace-Beltrami Operator:** Approximates this fundamental intrinsic operator using the heat kernel method, crucial for various applications like diffusion, smoothing, and spectral analysis on surfaces.
### Code snippets
```python
phi: ScalarField = ScalarField(m)

for i in range(m.points.shape[0]):
  coords: np.array = m.points[i]
  phi.set_value(np.sin(coords[0] * 2), i)

ambient_gradient: np.array = phi.compute_gradient()                             # (N, 3)
surface_gradient: np.array = phi.compute_surface_gradient()                     # (N, 3)
partial_derivatives: np.array = phi.compute_partial_derivatives()               # (N, 2)

laplacian: np.array = phi.compute_laplace_beltrami(t=HEAT_SCALE_LAPLACIAN)      # (N,)
```

## TODO
* Vector and Tensor fields
* Covariant Derivative
* Ricci tensor
* Higher dimensions manifolds

## Contributing
Please feel free to submit issues or pull requests.

## Dependencies
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* [NetworkX](https://github.com/networkx/networkx)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [Open3D](https://github.com/isl-org/Open3D)