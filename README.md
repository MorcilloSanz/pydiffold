# PyDiffold :earth_americas:
`PyDiffold` is a Python library for `differential geometry`. It provides tools for approximating `local differential structure` (tangent spaces, normal vectors, Riemannian metric tensors...), as well as discrete `differential operators` such as surface gradients and the Laplace–Beltrami operator, using only point cloud data.

The library is designed to facilitate numerical experimentation in geometry processing and geometric PDEs by operating directly on sampled surfaces, 
without requiring explicit mesh connectivity.

## Geodesics
A geodesic is the shortest path between two points on a curved surface or manifold. It generalizes the concept of a straight line to curved spaces. Geodesics represent the natural “straightest” paths that respect the manifold’s geometry

<p align="center">
    <img src="/img/geodesic.png" alt="Imagen 1" width="300"/><br/>
</p>

The arc length of a geodesic is the distance measured along the curve between its endpoints. It quantifies the actual length of the path on the manifold, taking into account its curvature. Calculating the arc length is essential for understanding the geometry and distances on curved surfaces:

$$L(\gamma )=\int _{a}^{b}{\sqrt {g_{\gamma (t)}({\dot {\gamma }}(t),{\dot {\gamma }}(t))}}\,dt$$

## PDE
Since `PyDiffold` allows you to define functions on a manifold and perform advanced differential calculations such as the gradient, surface gradient, and Laplace-Beltrami operator, it provides powerful tools to solve partial differential equations (PDEs) on curved spaces.

**Heat Equation**  
The heat equation governs the diffusion of a scalar field $\phi(x,y,z;t)$ over a Riemannian manifold $(M,g)$, where $\Delta$ denotes the Laplace-Beltrami operator. It is given by:

$$\frac{\partial \phi}{\partial t} = \alpha \Delta \phi$$

describing how heat dissipates over time according to the intrinsic geometry of the manifold.

**Wave Equation**  
The wave equation describes the propagation of waves in a Riemannian manifold $(M,g)$, modeling second-order hyperbolic dynamics of the scalar field $\phi(x,y,z;t)$. It is expressed as:

$$\frac{\partial^2 \phi}{\partial t^2} = c^2 \Delta \phi$$ 

where $c$ is the wave speed, and $\Delta$ denotes the Laplace-Beltrami operator, reflecting how curvature influences wave propagation.

**Heat and wave equations on a bunny surface**
<table align="center">
  <tr>
    <td align="center">
      <img src="/img/heat_equation.gif" alt="Imagen 1" width="300"/><br/>
      <small>Figure 1: Heat equation</small>
    </td>
    <td align="center" style="padding-left: 40px;">
      <img src="/img/wave_equation.gif" alt="Imagen 2" width="300"/><br/>
      <small>Figure 2: Wave equation</small>
    </td>
  </tr>
</table>

## Features :hammer_and_wrench:
* **Manifold graph:** computes a graph $G = (N,E)$ with associating point indices and distances.
* **Compute normals:** estimates normal vectors using PCA.
* **Compute tangent bundle:** computes the tangent bundle $TM$ using PCA.
* **Compute metric tensor:** computes the metric tensor $g_{\mu \nu}$ for each point $p$ of the manifold.
* **Compute geodesics:** computes the shortest path $\gamma(t)$ between two points of the manifold and its arc length $L$.
* **Define scalar fields in manifolds:** $\phi : \mathcal{M} \rightarrow \mathbb{R}$.
* **Compute gradient:** approximates the gradient $\nabla f$ of a scalar field defined in a manifold.
* **Compute surface gradient:** computes the surface gradient $\nabla_M f$ of a scalar field defined in a manifold.
* **Compute Laplace-Beltrami:** approximates the Laplace-Beltrami $\Delta f$ of a scalar field defined in a manifold.

## Dependencies
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* [NetworkX](https://github.com/networkx/networkx)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [Open3D](https://github.com/isl-org/Open3D)

## TODO
* Vector and Tensor fields
* Covariant Derivative
* Christoffel symbols
* Riemann Curvature Tensor
* Ricci tensor
* Higher dimensions manifolds