import numpy as np
import matplotlib.pyplot as plt

k_valores = np.arange(1, 11)

residuos32 = []
residuos64 = []

for k in k_valores:

    # float32
    a32 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b32 = np.array([1.0, 10.0**(-k), 0.0], dtype=np.float32)

    c32 = np.cross(a32, b32)
    r32 = abs(np.dot(a32, c32))

    residuos32.append(r32)

    # float64
    a64 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b64 = np.array([1.0, 10.0**(-k), 0.0], dtype=np.float64)

    c64 = np.cross(a64, b64)
    r64 = abs(np.dot(a64, c64))

    residuos64.append(r64)

    print(f"k = {k}")
    print(f"Residuo float32 = {r32}")
    print(f"Residuo float64 = {r64}")
    print()

# El valor cero no puede representarse en escala logaritmica.
# Se usa epsilon unicamente para poder visualizar los resultados.

eps32 = np.finfo(np.float32).eps
eps64 = np.finfo(np.float64).eps

grafica32 = np.maximum(residuos32, eps32)
grafica64 = np.maximum(residuos64, eps64)

plt.semilogy(k_valores, grafica32, 'o-', label='float32')
plt.semilogy(k_valores, grafica64, 's-', label='float64')

plt.xlabel('k')
plt.ylabel('Residuo |a . c|')
plt.title('Residuo de ortogonalidad')
plt.grid(True, which='both')
plt.legend()

plt.savefig('residuo_ortogonalidad.png',
            dpi=300,
            bbox_inches='tight')

plt.show()