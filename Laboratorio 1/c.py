import math


def sinc_maclaurin(x, numero_terminos):
    suma = 0.0

    for k in range(numero_terminos):
        termino = ((-1) ** k) * (x ** (2 * k))
        termino /= math.factorial(2 * k + 1)
        suma += termino

    return suma


x = 0.6
numero_terminos = 4

# Polinomio de grado 6
p6 = sinc_maclaurin(x, numero_terminos)

# Valor de referencia
valor_real = math.sin(x) / x

# Comparación con seis decimales
p6_redondeado = round(p6, 6)
valor_real_redondeado = round(valor_real, 6)

print("Número de términos:", numero_terminos)
print("Grado del polinomio:", 2 * (numero_terminos - 1))
print("P6(0.6):", p6)
print("sin(0.6)/0.6:", valor_real)
print("P6 redondeado a 6 decimales:", p6_redondeado)
print("Valor real redondeado a 6 decimales:",
      valor_real_redondeado)
print("¿Coinciden a 6 decimales?",
      p6_redondeado == valor_real_redondeado)