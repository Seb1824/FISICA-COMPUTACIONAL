"""Rendija simple: reciprocidad entre ancho y extension de la difraccion."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from lab7_optica import extension, fraunhofer, malla, normalizar

OUT = Path(__file__).resolve().parent


def main():
    lam, z, altura = 632.8e-9, 1.0, 200e-6
    n, lado = 1024, 2e-3
    dx = lado/n
    x = malla(n, lado)
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.6), layout='constrained')
    resultados = []
    xu = np.linspace(-24e-3, 24e-3, 1801)
    for fila, ancho in enumerate((100e-6, 50e-6)):
        columnas = abs(x) <= ancho/2
        apertura = (abs(x) <= altura/2)[:, None] & columnas[None, :]
        eje, campo = fraunhofer(apertura, dx, lam, z)
        intensidad = normalizar(abs(campo)**2)
        perfil = intensidad[n//2]
        primer_cero = lam*z/ancho
        for orden in (1, 2, 3):
            teorico = orden*primer_cero
            candidatos = np.flatnonzero(abs(eje-teorico) < .3*primer_cero)
            j = candidatos[np.argmin(perfil[candidatos])]
            resultados.append((ancho*1e6, columnas.sum()*dx*1e6, orden,
                               teorico*1e3, eje[j]*1e3))
        mapa, ax = axes[fila]
        mapa.imshow(intensidad, extent=extension(eje*1e3), origin='lower',
                    cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        mapa.set(xlim=(-24, 24), ylim=(-10, 10), xlabel='x (mm)',
                 ylabel='y (mm)', title=f'Ancho {ancho*1e6:g} µm')
        ax.plot(xu*1e3, np.sinc(ancho*xu/(lam*z))**2, label='Teoría')
        visibles = abs(eje) <= 24e-3
        ax.plot(eje[visibles]*1e3, perfil[visibles], '.', ms=3, label='FFT2D')
        ax.axvline(primer_cero*1e3, color='0.5', lw=.8, ls=':')
        ax.axvline(-primer_cero*1e3, color='0.5', lw=.8, ls=':')
        ax.set(xlim=(-24, 24), xlabel='x (mm)', ylabel=r'$I(x,0)/I_0$')
        ax.legend(loc='upper right')
    with (OUT/'lab7_punto1_minimos.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ancho_um', 'ancho_discreto_um', 'orden',
                    'minimo_teorico_mm', 'minimo_FFT_mm'])
        w.writerows(resultados)
    print(np.array(resultados))
    fig.savefig(OUT/'lab7_punto1.png', dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    main()
