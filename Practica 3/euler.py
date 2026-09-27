import numpy as np

# Datos
v0 = 25.0
theta = np.radians(40.0)
g = 9.81
T = 1.0

# Componentes iniciales
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

# Estado inicial
u0 = np.array([0.0, 0.0, vx0, vy0])


# Sistema dinamico
def F(u):
    x, y, vx, vy = u
    return np.array([vx, vy, 0.0, -g])


# Euler explicito
def euler(u0, h, T):
    u = u0.copy()
    n = int(round(T / h))

    for _ in range(n):
        u = u + h * F(u)

    return u

# RK2 / Heun
def rk2(u0, h, T):
    u = u0.copy()
    n = int(round(T / h))

    for _ in range(n):
        k1 = F(u)
        k2 = F(u + h * k1)

        u = u + (h / 2.0) * (k1 + k2)

    return u

# RK4
def rk4(u0, h, T):
    u = u0.copy()
    n = int(round(T / h))

    for _ in range(n):
        k1 = F(u)
        k2 = F(u + (h / 2.0) * k1)
        k3 = F(u + (h / 2.0) * k2)
        k4 = F(u + h * k3)

        u = u + (h / 6.0) * (
            k1 + 2*k2 + 2*k3 + k4
        )

    return u

pasos = [0.10, 0.05]

for h in pasos:

    u_euler = euler(u0, h, T)
    u_rk2 = rk2(u0, h, T)
    u_rk4 = rk4(u0, h, T)

    print(f"\nh = {h}")

    print("Euler:")
    print(u_euler)

    print("RK2/Heun:")
    print(u_rk2)

    print("RK4:")
    print(u_rk4)