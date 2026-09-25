import math

def logaritmo_maclaurin(x, numero_terminos):

    suma = 0.0  

    # Calculamos y sumamos cada término de la serie.
    for k in range(1, numero_terminos + 1):
        # (-1)^(k-1) hace que los signos alternen: +, -, +, ...
        termino = ((-1) ** (k - 1)) * (x ** k) / k
        suma += termino

    return suma


# Valor de x utilizado para aproximar
x = 0.15
numero_terminos = 5

# Aproximación mediante el polinomio de Maclaurin de quinto grado.
p5 = logaritmo_maclaurin(x, numero_terminos)

# Valor calculado directamente con la función logarítmica de Python.
valor_real = math.log(1 + x)

# Redondeamos ambos resultados para compararlos con cinco decimales.
p5_redondeado = round(p5, 5)
valor_real_redondeado = round(valor_real, 5)

print("Número de términos:", numero_terminos)
print("P5(0.15):", p5)
print("ln(1 + 0.15):", valor_real)
print("P5 redondeado a 5 decimales:", p5_redondeado)
print(
    "ln(1 + 0.15) redondeado a 5 decimales:",
    valor_real_redondeado
)

print(
    "¿Coinciden a 5 decimales?",
    p5_redondeado == valor_real_redondeado
)