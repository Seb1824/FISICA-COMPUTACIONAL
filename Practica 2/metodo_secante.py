P = 2.63e6       # Pa
T = 500.0        # K
a = 0.5536       # Pa*m^6/mol^2
b = 3.049e-5     # m^3/mol
R = 8.31446      # J/(mol*K)

tol = 1e-8

# Funcion de Van der Waals
def f(v):
    return P*v**3 - (P*b + R*T)*v**2 + a*v - a*b

# Metodo de la Secante
def secante(v0, v1, tol, max_iter=100):
    iteraciones = 0

    for i in range(max_iter):

        f0 = f(v0)
        f1 = f(v1)

        if f1 - f0 == 0:
            print("Division entre cero.")
            return None, iteraciones

        v2 = v1 - f1 * (v1 - v0) / (f1 - f0)

        iteraciones += 1

        if abs(v2 - v1) < tol:
            return v2, iteraciones

        v0 = v1
        v1 = v2

    return v2, iteraciones


# FASE LIQUIDA
v0_liq = 1.1 * b
v1_liq = 1.2 * b

vl, iter_liq = secante(v0_liq, v1_liq, tol)

# FASE GASEOSA
v0_gas = R * T / P
v1_gas = 0.9 * v0_gas

vg, iter_gas = secante(v0_gas, v1_gas, tol)

print("FASE LIQUIDA")
print("Raiz =", vl)
print("Iteraciones =", iter_liq)
print("Residuo =", abs(f(vl)))
print()
print("FASE GASEOSA")
print("Raiz =", vg)
print("Iteraciones =", iter_gas)
print("Residuo =", abs(f(vg)))