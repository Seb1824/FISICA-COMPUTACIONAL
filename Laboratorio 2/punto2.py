# Constantes del CO2
P = 1.0e6       # Pa
T = 300.0       # K
a = 0.3640      # Pa*m^6/mol^2
b = 4.267e-5    # m^3/mol
R = 8.31446     # J/(mol*K)

# Funcion de Van der Waals
def f(v):
    return P*v**3 - (P*b + R*T)*v**2 + a*v - a*b

# Derivada de la funcion
def df(v):
    return 3*P*v**2 - 2*(P*b + R*T)*v + a


# Tolerancia
tol = 1e-8


# Metodo de Biseccion
def biseccion(a_int, b_int, tol):
    iteraciones = 0

    while (b_int - a_int) / 2 > tol:
        c = (a_int + b_int) / 2

        if f(a_int) * f(c) < 0:
            b_int = c
        else:
            a_int = c

        iteraciones += 1

    raiz = (a_int + b_int) / 2

    return raiz, iteraciones


# Metodo de Newton-Raphson
def newton_raphson(v0, tol):
    iteraciones = 0
    v = v0

    while True:
        v_nuevo = v - f(v) / df(v)

        iteraciones += 1

        if abs(v_nuevo - v) < tol:
            break

        v = v_nuevo

    return v_nuevo, iteraciones


# Intervalo para Biseccion
v_min = b + 1e-6
v_max = 0.01

# Valor inicial para Newton-Raphson
v0 = R * T / P


# Ejecutar Biseccion
raiz_bis, iter_bis = biseccion(v_min, v_max, tol)

# Ejecutar Newton-Raphson
raiz_newton, iter_newton = newton_raphson(v0, tol)


# Mostrar resultados
print("METODO DE BISECCION")
print("Raiz =", raiz_bis)
print("Iteraciones =", iter_bis)
print("Residuo =", abs(f(raiz_bis)))

print()

print("METODO DE NEWTON-RAPHSON")
print("Raiz =", raiz_newton)
print("Iteraciones =", iter_newton)
print("Residuo =", abs(f(raiz_newton)))