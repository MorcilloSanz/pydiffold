# PyDiffold :earth_americas:
`PyDiffold` is a Python library for `differential geometry`. It provides tools for approximating `local differential structure` (tangent spaces, normal vectors, Riemannian metric tensors...), as well as discrete `differential operators` such as surface gradients and the Laplace–Beltrami operator, using only point cloud data.

The library is designed to facilitate numerical experimentation in geometry processing and geometric PDEs by operating directly on sampled surfaces, 
without requiring explicit mesh connectivity.

### PyDiffold for Partial Differential Equations
The Laplace–Beltrami operator was computed on the vertices of the Stanford Bunny, without relying on mesh connectivity. Based on this operator, the heat equation and the wave equation were solved on the surface of the model.

The resulting solutions are shown as GIF animations, illustrating the diffusion of heat and the propagation of waves over the geometry.

<p align="center">
  <img src="/img/heat_equation_1.gif" width="200"/>
  <img src="/img/heat_equation_2.gif" width="200"/>
  <img src="/img/wave_equation_1.gif" width="200"/>
  <img src="/img/wave_equation_2.gif" width="200"/>
</p>

## Local differential structure

**Compute 2-manifold:** computes the manifold $M$. Defined by an array of 3D points (N, 3), that represents a discretized surface.
```python
points: np.array = np.loadtxt('/assets/bunny.txt')
manifold: Manifold = Manifold(points)
```

**Normal bundle:** computes the normal bundle $NM$ of the manifold $M$ (N, 3).
```python
normal_bundle: np.array = manifold.normal_bundle
```

**Tangent bundle:** computes the tangent bundle $TM$ of the manifold $M$ (N, 2, 3).
```python
tangent_bundle: np.array = manifold.tangent_bundle
```

**Metric tensor:** computes the metric tensor $g_{\mu \nu}$, its inverse $g^{\mu \nu}$ (N, 2, 2); and its derivatives $\partial_{\mu} g_{\mu \nu}$ and $\partial_{\nu} g_{\mu \nu}$ (N, 2, 2, 2) for each point $p \in M$.
```python
metric_tensor: np.array = manifold.metric_tensor
metric_tensor_inv: np.array = manifold.metric_tensor_inv
metric_tensor_derivatives: np.array = manifold.metric_tensor_derivatives
```

**Christoffel Symbols:** computes the Christoffel Symbols $\Gamma^{\sigma}_{\mu \nu}$ (N, 2, 2, 2) and its derivatives $\partial_{\mu} \Gamma^{\sigma}_{\mu \nu}$ and $\partial_{\nu} \Gamma^{\sigma}_{\mu \nu}$ (N, 2, 2, 2, 2) for each point $p \in M$.
```python
christoffel_symbols: np.array = manifold.christoffel_symbols
christoffel_symbols_deivatives: np.array = manifold.christoffel_symbols_derivatives
```

**Riemann curvature tensor:** computes the Riemann curvature tensor $R^{\rho}_{\sigma \mu \nu}$ (N, 2, 2, 2, 2) for each point $p \in M$.
```python
riemann_tensor: np.array = manifold.riemann_tensor
```

**Compute geodesics:** computes the shortest path $\gamma(t)$ between two points of the manifold and its arc length $L$.
```python
geodesic, arc_length = manifold.geodesic(0, 2000)
geodesic_coords: np.array = manifold.points[geodesic]
```

## Differential operators

**Define scalar field:** define a scalar field $\phi : M \rightarrow \mathbb{R}$ that associates each point of the manifold $p \in M$ with a scalar.
```python
phi: ScalarField = ScalarField(manifold)

for i in range(manifold.points.shape[0]):
  coords: np.array = manifold.points[i]
  phi.set_value(np.sin(coords[0] * 2), i)
```

**Compute (ambient) gradient:** computes the gradient $\nabla \phi$ (N, 3) of a scalar field  for each point $p \in M$.
```python
ambient_gradient: np.array = phi.compute_gradient()
```

**Compute surface gradient:** computes the surface gradient $\nabla_M \phi$ (N, 3) of a scalar field  for each point $p \in M$.
```python
surface_gradient: np.array = phi.compute_surface_gradient()
```

**Compute partial derivatives in the tangent directions (directional derivatives)**: computes the partial derivatives $\partial_{\mu} \phi$ and $\partial_{\nu} \phi$ (N, 2) for each point $p \in M$.
```python
partial_derivatives: np.array = function.compute_partial_derivatives()
```

**Compute Laplace-Beltrami:** computes the Laplace-Beltrami $\Delta_M \phi$ of a scalar field for each point $p \in M$.
```python
laplacian: np.array = phi.compute_laplace_beltrami(t=HEAT_SCALE_LAPLACIAN)
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