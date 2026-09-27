import math
import numpy as np

# Importar los metodos definidos en euler.py
from euler import euler, rk2, rk4

# Datos del problema
v0 = 25.0
theta = np.radians(40.0)
g = 9.81
T = 1.0

# Componentes iniciales
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

# Estado inicial
u0 = np.array([0.0, 0.0, vx0, vy0])

# Solucion exacta
x_exacta = v0 * np.cos(theta) * T
y_exacta = v0 * np.sin(theta) * T - 0.5 * g * T**2

r_exacta = np.array([x_exacta, y_exacta])


# Error de posicion
def error_posicion(u):
    r_numerica = u[:2]
    return np.linalg.norm(r_numerica - r_exacta)


# Orden empirico
def orden(Eh, Eh2):
    if Eh < 1e-12 or Eh2 < 1e-12:
        return None

    return math.log(Eh / Eh2, 2)


# Metodos a comparar
metodos = {
    "Euler": euler,
    "RK2": rk2,
    "RK4": rk4
}


# Calculo para h = 0.10 y h/2 = 0.05
for nombre, metodo in metodos.items():

    u_h = metodo(u0, 0.10, T)
    u_h2 = metodo(u0, 0.05, T)

    E_h = error_posicion(u_h)
    E_h2 = error_posicion(u_h2)

    p = orden(E_h, E_h2)

    print(nombre)
    print("E(0.10) =", E_h)
    print("E(0.05) =", E_h2)

    if p is None:
        print("Orden p = no evaluable numericamente")
    else:
        print("Orden p =", p)

    print()