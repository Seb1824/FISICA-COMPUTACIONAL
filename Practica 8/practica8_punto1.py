"""Transformacion de velocidades y supervivencia de muones."""
from pathlib import Path
from math import exp, sqrt

OUT = Path(__file__).resolve().parent
C = 299_792_458.0


def main():
    beta_a, beta_b = 0.8, -0.8
    beta_gal = beta_b-beta_a
    beta_lor = beta_gal/(1-beta_a*beta_b)
    altura, beta, vida = 10_000.0, 0.995, 2.20e-6
    dt = altura/(beta*C)
    gamma = 1/sqrt(1-beta**2)
    dtau = dt/gamma
    resultados = (
        ('velocidad_Galileo_sobre_c', beta_gal),
        ('velocidad_Lorentz_sobre_c', beta_lor),
        ('velocidad_Galileo_m_s', beta_gal*C),
        ('velocidad_Lorentz_m_s', beta_lor*C),
        ('tiempo_Tierra_us', dt*1e6),
        ('gamma', gamma),
        ('tiempo_propio_us', dtau*1e6),
        ('fraccion_relativista', exp(-dtau/vida)),
        ('fraccion_clasica', exp(-dt/vida)),
    )
    texto = ''.join(f'{nombre}: {valor:.12g}\n' for nombre, valor in resultados)
    (OUT/'practica8_punto1_resultados.txt').write_text(texto, encoding='utf-8')
    print(texto)


if __name__ == '__main__':
    main()
