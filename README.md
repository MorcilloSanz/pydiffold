# PyDiffold
PyDiffold is a Python library for `differential geometry` on 2D manifolds embedded in $\mathbb{R}^3$. It provides tools for approximating `local differential structure` (tangent spaces, normal vectors, Riemannian metric tensors...), as well as discrete `differential operators` such as surface gradients and the Laplace–Beltrami 
operator, using only point cloud data.

![](/img/fun.png)

The library is designed to facilitate numerical experimentation in geometry processing and geometric PDEs by operating directly on sampled surfaces, 
without requiring explicit mesh connectivity.

## Heat Equation

[Laplacian Eigenmaps for Dimensionality Reduction and Data Representation](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf)

The heat equation on a manifold is a fundamental geometric PDE that describes the diffusion of a scalar field $\phi(x,y,z;t)$ over time along the surface. In this example, PyDiffold solves the heat equation:

$$\frac{\partial \phi}{\partial t} = \alpha \Delta \phi$$

on a curved 2D manifold embedded in $\mathbb{R}^3$, where $\Delta$ denotes the Laplace–Beltrami operator. The simulation demonstrates the evolution of an initial heat distribution under intrinsic surface diffusion, with the geometry taken into account through local differential operators estimated from the point cloud.

The visualization shows how PyDiffold can perform time-dependent simulations of PDEs on manifolds without requiring a mesh, relying instead on a purely point-based representation.

![](/img/heat_equation.gif)

## Features
* **Manifold graph:** computes a graph $G = (N,E)$ with information about points, such as, indices, coordinates and distances using a KDTree.
* **Compute normals:** estimates normal vectors using PCA for each point $p$ of the manifold.
* **Compute tangent bundle:** computes the tangent bundle $TM$ estimating each tangent space $T_pM$ basis vectors using PCA for each point $p$ of the manifold.
* **Compute metric tensor:** computes the metric tensor $g_{\mu \nu}$ for each point $p$ of the manifold.
* **Compute geodesics:** computes the shortest path $\gamma(t)$ between two points of the manifold and its arc length $L$.
* **Define scalar fields in manifolds:** $f : \mathcal{M} \rightarrow \mathbb{R}$.
* **Compute gradient:** approximates the gradient $\nabla f$ of a scalar field defined in a manifold.
* **Compute surface gradient:** computes the surface gradient $\nabla_M f$ of a scalar field defined in a manifold.
* **Compute Laplace-Beltrami:** approximates the Laplace-Beltrami $\Delta f$ of a scalar field defined in a manifold.

## Dependencies
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* [NetworkX](https://github.com/networkx/networkx)
* [Matplotlib](https://github.com/matplotlib/matplotlib)

## TODO
* Vector and Tensor fields
* Covariant Derivative
* Christoffel symbols
* Riemann Curvature Tensor
* Ricci tensor
* Higher dimensions manifolds