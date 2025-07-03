# PyDiffold :earth_americas:
`PyDiffold` is a Python library for `differential geometry`. It provides tools for approximating `local differential structure` (tangent spaces, normal vectors, Riemannian metric tensors...), as well as discrete `differential operators` such as surface gradients and the Laplace–Beltrami operator, using only point cloud data.

The library is designed to facilitate numerical experimentation in geometry processing and geometric PDEs by operating directly on sampled surfaces, 
without requiring explicit mesh connectivity.

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
This code constructs a discrete `2D Riemannian manifold` $M$ from a 3D point cloud and computes its intrinsic geometric structure. It extracts fundamental differential geometry objects such as:
* **Tangent and normal bundles** $NM$ and $TM$ of $M$.
* **Metric tensor**, its inverse and partial derivatives, $g_{\mu \nu}, \, g^{\mu \nu}, \, \partial_{\mu} g_{\mu \nu}$ and $\partial_{\nu} g_{\mu \nu}$ for each point $p \in M$.
* **Christoffel Symbols** and its partial derivatives $\Gamma^{\sigma}_{\mu \nu}, \, \partial_{\mu} \Gamma^{\sigma}_{\mu \nu}$ and $\partial_{\nu} \Gamma^{\sigma}_{\mu \nu}$ for each point $p \in M$.
* **Riemann curvature tensor** $R^{\rho}_{\sigma \mu \nu}$ for each point $p \in M$.
```python
points: np.array = np.loadtxt('/assets/bunny.txt')                              # (N, 3)
manifold: Manifold = Manifold(points)

normal_bundle: np.array = manifold.normal_bundle                                # (N, 3)
tangent_bundle: np.array = manifold.tangent_bundle                              # (N, 2, 3)

metric_tensor: np.array = manifold.metric_tensor                                # (N, 2, 2)
metric_tensor_inv: np.array = manifold.metric_tensor_inv                        # (N, 2, 2)
metric_tensor_derivatives: np.array = manifold.metric_tensor_derivatives        # (N, 2, 2, 2)

chris_symbols: np.array = manifold.christoffel_symbols                          # (N, 2, 2, 2)
chris_symbols_deivatives: np.array = manifold.christoffel_symbols_derivatives   # (N, 2, 2, 2, 2)

riemann_tensor: np.array = manifold.riemann_tensor                              # (N, 2, 2, 2, 2)
```

**Compute geodesics:** computes the shortest path $\gamma(t)$ between two points of the manifold and its arc length $L$.
```python
geodesic, arc_length = manifold.geodesic(0, 2000)                               # (K,)
geodesic_coords: np.array = manifold.points[geodesic]                           # (K, 3)
```

## Differential operators
This code defines a `scalar field` on a Riemannian manifold and computes its `differential properties`. It evaluates the field over the surface, then calculates its ambient and intrinsic gradients, partial derivatives, and the Laplace–Beltrami operator, enabling geometric or physical analysis on the manifold.

* **Scalar field:** define a scalar field $\phi : M \rightarrow \mathbb{R}$.
* **Ambient gradient** of a scalar field $\nabla \phi$ for each point $p \in M$.
* **Surface gradient** of a scalar field $\nabla_M \phi$ for each point $p \in M$.
* **Directional derivatives** in the tangent directions $\partial_{\mu} \phi$ and $\partial_{\nu} \phi$ for each point $p \in M$.
* **Laplace-Beltrami** $\Delta_M \phi$ for each point $p \in M$.
```python
phi: ScalarField = ScalarField(manifold)

for i in range(manifold.points.shape[0]):
  coords: np.array = manifold.points[i]
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

## Dependencies
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* [NetworkX](https://github.com/networkx/networkx)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [Open3D](https://github.com/isl-org/Open3D)