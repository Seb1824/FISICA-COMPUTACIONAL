"""Disco de Airy y balance de Parseval con intensidades sin normalizar."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from practica7_optica import extension, fraunhofer, malla, normalizar

OUT = Path(__file__).resolve().parent


def main():
    n, lado, lam, z, radio = 1024, 3e-3, 632.8e-9, 1., 50e-6
    dx = lado/n
    x = malla(n, lado)
    a = x[:, None]**2 + x[None, :]**2 <= radio**2
    eje, u = fraunhofer(a, dx, lam, z)
    energia_ap = np.count_nonzero(a)*dx**2
    dx_pantalla = lam*z/lado
    intensidad = abs(u)**2
    energia_pant = intensidad.sum()*dx_pantalla**2
    error = abs(energia_ap-energia_pant)/energia_ap
    datos = (energia_ap, energia_pant, error)
    print(datos)
    with (OUT/'practica7_punto3_parseval.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['E_apertura', 'E_pantalla', 'error_relativo'])
        w.writerow(datos)

    i = normalizar(intensidad)
    radio_airy = 1.22*lam*z/(2*radio)
    fig, ejes = plt.subplots(1, 2, figsize=(9.5, 4), layout='constrained')
    for ax, mapa, titulo, etiqueta in zip(
        ejes, (i, np.log10(i+1e-6)), ('Lineal', 'Logarítmica'),
        (r'$I/I_0$', r'$\log_{10}(I/I_0+10^{-6})$')
    ):
        im = ax.imshow(mapa, extent=extension(eje*1e3), cmap='magma',
                       origin='lower', interpolation='nearest')
        ax.add_patch(plt.Circle((0, 0), radio_airy*1e3, fill=False,
                                color='#72e8ff', ls='--', lw=0.8))
        ax.set(xlim=(-25, 25), ylim=(-25, 25), title=titulo,
               xlabel='x (mm)', ylabel='y (mm)')
        fig.colorbar(im, ax=ax, label=etiqueta, shrink=.82)
    fig.savefig(OUT/'practica7_punto3.png', dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    main()
