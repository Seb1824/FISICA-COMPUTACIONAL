import math


def coseno_maclaurin(x, numero_terminos):
    suma = 0.0

    for k in range(numero_terminos):
        termino = ((-1) ** k) * (x ** (2 * k))
        termino /= math.factorial(2 * k)
        suma += termino

    return suma


x = 0.4
numero_terminos = 4

# Evaluar la suma truncada P6(0.4)
p6 = coseno_maclaurin(x, numero_terminos)

# Obtener el valor de referencia
valor_real = math.cos(x)

# Redondear ambos valores a cinco decimales
p6_redondeado = round(p6, 5)
valor_real_redondeado = round(valor_real, 5)

print("Número de términos:", numero_terminos)
print("P6(0.4):", p6)
print("cos(0.4):", valor_real)
print("P6 redondeado a 5 decimales:", p6_redondeado)
print("cos(0.4) redondeado a 5 decimales:", valor_real_redondeado)
print("¿Coinciden a 5 decimales?", p6_redondeado == valor_real_redondeado)