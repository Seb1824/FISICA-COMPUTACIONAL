import math

x = 0.4

# Polinomio de Maclaurin con tres términos
p4 = 1 - (x**2 / math.factorial(2)) + (x**4 / math.factorial(4))

aproximacion = round(p4, 4)
valor_real = round(math.cos(x), 4)

print("P4(0.4):", p4)
print("cos(0.4):", math.cos(x))
print("P4 redondeado a 4 decimales:", aproximacion)
print("cos(0.4) redondeado a 4 decimales:", valor_real)
