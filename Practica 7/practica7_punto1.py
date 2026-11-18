"""Rendija rectangular: FFT2D y comparacion con sinc al cuadrado."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from practica7_optica import extension, fraunhofer, malla, normalizar

OUT = Path(__file__).resolve().parent


def main():
    lam, z, ancho, alto = 632.8e-9, 1.5, 80e-6, 160e-6
    n, lado = 1024, 4e-3
    dx = lado / n
    x = malla(n, lado)
    apertura = (abs(x) <= alto/2)[:, None] & (abs(x) <= ancho/2)[None, :]
    eje, u = fraunhofer(apertura, dx, lam, z)
    i = normalizar(abs(u)**2)
    curva = i[n//2]
    teoricos = np.arange(1, 4) * lam*z/ancho
    filas = []
    for orden, xm in enumerate(teoricos, 1):
        candidatos = np.flatnonzero(abs(eje - xm) < 0.3*teoricos[0])
        j = candidatos[np.argmin(curva[candidatos])]
        filas.append((orden, 1e3*xm, 1e3*eje[j], 1e3*abs(eje[j]-xm)))
    with (OUT/'practica7_punto1_minimos.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['orden', 'teoria_mm', 'FFT_mm', 'error_absoluto_mm'])
        w.writerows(filas)
    print(f'Ancho representado: {apertura[n//2].sum()*dx*1e6:.6f} um')
    print(np.array(filas))

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    a.imshow(i, extent=extension(eje*1e3), origin='lower', cmap='magma',
             interpolation='nearest')
    a.set(xlim=(-40, 40), ylim=(-40, 40), xlabel='x (mm)', ylabel='y (mm)',
          title='Intensidad 2D normalizada')
    xu = np.linspace(-40e-3, 40e-3, 2001)
    b.plot(xu*1e3, np.sinc(ancho*xu/(lam*z))**2, label='Teoría')
    b.plot(eje*1e3, curva, '.', ms=3, label='FFT2D')
    for _, _, pos, _ in filas:
        b.axvline(pos, color='0.5', ls=':', lw=0.8)
    b.set(xlim=(-40, 40), xlabel='x (mm)', ylabel=r'$I(x,0)/I_0$',
          title='Corte transversal')
    b.legend()
    fig.savefig(OUT/'practica7_punto1.png', dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    main()
