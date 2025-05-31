# PyDiffold
PyDiffold is a Python library for differential geometric computations on 2D manifolds embedded in $\mathbb{R}^3$. It provides tools for approximating local differential 
structure (tangent spaces, normal vectors, Riemannian metric tensors...), as well as discrete differential operators such as surface gradients and the Laplace–Beltrami 
operator, using only point cloud data.

The library is designed to facilitate numerical experimentation in geometry processing and geometric PDEs by operating directly on sampled surfaces, 
without requiring explicit mesh connectivity.

[Laplacian Eigenmaps for Dimensionality Reduction and Data Representation](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf)

## Heat Equation Example

The heat equation on a manifold is a fundamental geometric PDE that describes the diffusion of a scalar field \( \phi \) over time along the surface. In this example, PyDiffold solves the heat equation

$$\frac{\partial \phi}{\partial t} = \alpha \Delta \phi$$

on a curved 2D manifold embedded in $\mathbb{R}^3$, where $\Delta$ denotes the Laplace–Beltrami operator. The simulation demonstrates the evolution of an initial heat distribution under intrinsic surface diffusion, with the geometry taken into account through local differential operators estimated from the point cloud.

The visualization shows how PyDiffold can perform time-dependent simulations of PDEs on manifolds without requiring a mesh, relying instead on a purely point-based representation.

![](/img/heat_equation.gif)

## Features
* Compute normals
* Compute tangent bundle
* Compute normal bundle
* Compute metric tensor
* Compute geodesics
* Define scalar fields in manifolds
* Compute gradient
* Compute surface gradient
* Compute Laplace-Beltrami


## TODO
* Vector and Tensor fields
* Covariant Derivative
* Christoffel symbols
* Riemann Curvature Tensor
* Ricci tensor