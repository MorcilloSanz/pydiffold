import numpy as np

if __name__ == "__main__":
    # Rango de la rejilla
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x, y)

    # Superficie: paraboloide hiperbólico
    z = x**2 - y**2

    # Aplanar las matrices y apilar los vectores columna
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Guardar en archivo txt
    np.savetxt("surface.txt", points, fmt="%.6f")