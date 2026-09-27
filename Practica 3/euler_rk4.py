import numpy as np
import matplotlib.pyplot as plt

# Constante gravitacional combinada GM
GM = 3.986e14       # m^3/s^2

# Condiciones iniciales
r0 = 7.0e6          # m
v0 = np.sqrt(GM / r0)

u0 = np.array([
    r0,
    0.0,
    0.0,
    v0
])


# Funcion dinamica
def F(u):
    x, y, vx, vy = u

    r = np.sqrt(x**2 + y**2)

    ax = -GM * x / r**3
    ay = -GM * y / r**3

    return np.array([
        vx,
        vy,
        ax,
        ay
    ])


# Metodo de Euler-Cromer
def euler_cromer(u0, h, n):

    u = u0.copy()
    estados = [u.copy()]

    for _ in range(n):

        x, y, vx, vy = u

        r = np.sqrt(x**2 + y**2)

        ax = -GM * x / r**3
        ay = -GM * y / r**3

        # Primero se actualiza la velocidad
        vx = vx + h * ax
        vy = vy + h * ay

        # Luego se actualiza la posicion
        x = x + h * vx
        y = y + h * vy

        u = np.array([x, y, vx, vy])

        estados.append(u.copy())

    return np.array(estados)


# Metodo RK4
def rk4(u0, h, n):

    u = u0.copy()
    estados = [u.copy()]

    for _ in range(n):

        k1 = F(u)
        k2 = F(u + h*k1/2)
        k3 = F(u + h*k2/2)
        k4 = F(u + h*k3)

        u = u + (h/6) * (
            k1 + 2*k2 + 2*k3 + k4
        )

        estados.append(u.copy())

    return np.array(estados)

T = 2 * np.pi * np.sqrt(r0**3 / GM)

h = 20.0

n = int(T / h)

estados_ec = euler_cromer(u0, h, n)
estados_rk4 = rk4(u0, h, n)

tiempo = np.arange(n + 1) * h

def energia(estados):

    x = estados[:, 0]
    y = estados[:, 1]

    vx = estados[:, 2]
    vy = estados[:, 3]

    r = np.sqrt(x**2 + y**2)

    return (
        0.5 * (vx**2 + vy**2)
        - GM / r
    )


energia_ec = energia(estados_ec)
energia_rk4 = energia(estados_rk4)


error_ec = np.abs(
    (energia_ec - energia_ec[0])
    / energia_ec[0]
)

error_rk4 = np.abs(
    (energia_rk4 - energia_rk4[0])
    / energia_rk4[0]
)


delta_ec = np.max(error_ec)
delta_rk4 = np.max(error_rk4)


# Trayectoria orbital
plt.figure(figsize=(7, 7))

plt.plot(
    estados_ec[:, 0],
    estados_ec[:, 1],
    label="Euler-Cromer"
)

plt.plot(
    estados_rk4[:, 0],
    estados_rk4[:, 1],
    label="RK4"
)

plt.scatter(
    [0],
    [0],
    label="Centro gravitacional"
)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Trayectoria orbital")
plt.axis("equal")
plt.grid()
plt.legend()

plt.savefig(
    "trayectoria_orbital.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Error relativo de energia
plt.figure(figsize=(8, 5))

plt.plot(
    tiempo,
    error_ec,
    label="Euler-Cromer"
)

plt.plot(
    tiempo,
    error_rk4,
    label="RK4"
)

plt.xlabel("Tiempo [s]")
plt.ylabel("Error relativo de energia")
plt.title("Conservacion de energia")
plt.grid()
plt.legend()

plt.savefig(
    "error_energia.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("Euler-Cromer:")
print("Error maximo =", delta_ec)

print()

print("RK4:")
print("Error maximo =", delta_rk4) 

print("Periodo orbital =", T, "s")
print("Numero de pasos =", n)
print("Tiempo simulado =", n*h, "s")