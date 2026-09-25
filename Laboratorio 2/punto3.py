import numpy as np
import matplotlib.pyplot as plt

# Constantes del CO2
P = 1.0e6       # Pa
T = 300.0       # K
a = 0.3640      # Pa*m^6/mol^2
b = 4.267e-5    # m^3/mol
R = 8.31446     # J/(mol*K)

tol = 1e-8


# Funcion de Van der Waals
def f(v):
    return P*v**3 - (P*b + R*T)*v**2 + a*v - a*b


# Derivada
def df(v):
    return 3*P*v**2 - 2*(P*b + R*T)*v + a


# Metodo de Biseccion
def biseccion(a_int, b_int, tol):
    aproximaciones = []

    while (b_int - a_int) / 2 > tol:
        c = (a_int + b_int) / 2
        aproximaciones.append(c)

        if f(a_int) * f(c) < 0:
            b_int = c
        else:
            a_int = c

    raiz = (a_int + b_int) / 2

    return raiz, aproximaciones


# Metodo de Newton-Raphson
def newton_raphson(v0, tol):
    aproximaciones = []
    v = v0

    while True:
        v_nuevo = v - f(v) / df(v)
        aproximaciones.append(v_nuevo)

        if abs(v_nuevo - v) < tol:
            break

        v = v_nuevo

    return v_nuevo, aproximaciones


# Intervalo de Biseccion
v_min = b + 1e-6
v_max = 0.01

# Valor inicial de Newton-Raphson
v0 = R * T / P


# Ejecutar metodos
raiz_bis, aprox_bis = biseccion(v_min, v_max, tol)
raiz_newton, aprox_newton = newton_raphson(v0, tol)


# Se toma la raiz de Newton-Raphson como referencia
# por su alta precision
raiz_ref = raiz_newton


# Calcular errores por iteracion
errores_bis = [
    abs(v - raiz_ref)
    for v in aprox_bis
]

errores_newton = [
    abs(v - raiz_ref)
    for v in aprox_newton
]

# Evitar valores exactamente cero en escala logaritmica
errores_bis = np.maximum(errores_bis, np.finfo(float).eps)
errores_newton = np.maximum(errores_newton, np.finfo(float).eps)


# GRAFICAS
fig, ax = plt.subplots(1, 2, figsize=(13, 5))


# GRAFICA 1: Funcion f(v)
v_valores = np.linspace(v_min, 0.004, 500)
f_valores = f(v_valores)

ax[0].plot(v_valores, f_valores, label="f(v)")
ax[0].axhline(0, color="black", linewidth=0.8)

ax[0].plot(
    raiz_newton,
    f(raiz_newton),
    "ro",
    label=f"Raiz = {raiz_newton:.8f}"
)

ax[0].set_xlabel("Volumen molar v [m^3/mol]")
ax[0].set_ylabel("f(v)")
ax[0].set_title("Funcion de Van der Waals")
ax[0].grid(True)
ax[0].legend()


# GRAFICA 2: Convergencia
iter_bis = range(1, len(errores_bis) + 1)
iter_newton = range(1, len(errores_newton) + 1)

ax[1].semilogy(
    iter_bis,
    errores_bis,
    "o-",
    label="Biseccion"
)

ax[1].semilogy(
    iter_newton,
    errores_newton,
    "s-",
    label="Newton-Raphson"
)

ax[1].set_xlabel("Iteracion")
ax[1].set_ylabel("Error |v_n - v*|")
ax[1].set_title("Velocidad de convergencia")
ax[1].grid(True, which="both")
ax[1].legend()


plt.tight_layout()

plt.savefig(
    "convergencia_metodos.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("METODO DE BISECCION")
print("Raiz =", raiz_bis)
print("Iteraciones =", len(aprox_bis))
print()
print("METODO DE NEWTON-RAPHSON")
print("Raiz =", raiz_newton)
print("Iteraciones =", len(aprox_newton))