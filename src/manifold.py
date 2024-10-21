import numpy as np

class Manifold:

    def __init__(self, x_mesh: np.array, y_mesh: np.array, z_mesh: np.array) -> None:
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.z_mesh = z_mesh

    @property
    def x(self) -> np.array:
        return self.x_mesh.flatten()
    
    @property
    def y(self) -> np.array:
        return self.y_mesh.flatten()
    
    @property
    def z(self) -> np.array:
        return self.z_mesh.flatten()