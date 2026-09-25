# Constantes del CO2
P = 1.0e6       # Pa
T = 300.0       # K
a = 0.3640      # Pa*m^6/mol^2
b = 4.267e-5    # m^3/mol
R = 8.31446     # J/(mol*K)

# Ecuacion de Van der Waals
def f(v):
    return P*v**3 - (P*b + R*T)*v**2 + a*v - a*b

# Derivada de f(v)
def df(v):
    return 3*P*v**2 - 2*(P*b + R*T)*v + a

v = 0.002

print("f(v) =", f(v))
print("f'(v) =", df(v))