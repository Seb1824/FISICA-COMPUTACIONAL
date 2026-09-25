import numpy as np

a = np.array([1.0, 0.0, 0.0])

for k in range(1, 11):
    b = np.array([1.0, 10.0**(-k), 0.0])

    c = np.cross(a, b)
    ortogonalidad = np.dot(a, c)

    print(f"k = {k}")
    print("c =", c)
    print("a . c =", ortogonalidad)
    print()