# PyDiffold
`PyDiffold` is a Python library for `differential geometry` on 2D manifolds embedded in $\mathbb{R}^3$. It provides tools for approximating `local differential structure` (tangent spaces, normal vectors, Riemannian metric tensors...), as well as discrete `differential operators` such as surface gradients and the Laplace–Beltrami 
operator, using only point cloud data.

The library is designed to facilitate numerical experimentation in geometry processing and geometric PDEs by operating directly on sampled surfaces, 
without requiring explicit mesh connectivity.

## PDE
Since `PyDiffold` allows you to define a function as a scalar field, consider a scalar field $\phi(x,y,z;t)$ is defined on a manifold $M$. In the test folder, you can find example code for solving the heat and wave equations on a manifold using PyDiffold computing the Laplace-Beltrami operator.

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

**Heat Equation:**  
The heat equation models the temporal evolution of temperature \(\phi\) in a medium, describing heat diffusion:  
$$
\frac{\partial \phi}{\partial t} = \alpha \Delta \phi
$$

**Wave Equation:**  
The wave equation models the propagation of disturbances \(\phi\) in a medium, such as sound or electromagnetic waves:  
$$
\frac{\partial^2 \phi}{\partial t^2} = c^2 \Delta \phi
$$

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

## References
[Laplacian Eigenmaps for Dimensionality Reduction and Data Representation](https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf)