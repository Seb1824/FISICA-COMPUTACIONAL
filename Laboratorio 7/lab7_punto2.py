"""Doble rendija y red de seis rendijas."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from lab7_optica import extension, fraunhofer, malla, normalizar

OUT = Path(__file__).resolve().parent


def apertura(x, ancho, altura, separacion, m):
    centros = (np.arange(m) - (m-1)/2) * separacion
    columnas = np.any(abs(x[:, None] - centros) <= ancho/2, axis=1)
    return (abs(x) <= altura/2)[:, None] & columnas[None, :]


def main():
    lam, z = 532e-9, 2.0
    ancho, d, altura = 60e-6, 240e-6, 2e-3
    n, lado = 1024, 4e-3
    x = malla(n, lado)
    perfiles = {}
    for m in (2, 6):
        a = apertura(x, ancho, altura, d, m)
        eje, u = fraunhofer(a, lado/n, lam, z)
        perfiles[m] = normalizar(abs(u)**2)[n//2]
        if m == 2:
            doble = a
    xd = np.linspace(-22e-3, 22e-3, 2001)
    envolvente = np.sinc(ancho*xd/(lam*z))**2
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4), layout='constrained')
    a.imshow(doble, extent=extension(x*1e3), origin='lower',
             cmap='Blues', interpolation='nearest', aspect='auto')
    a.set(xlim=(-.35, .35), ylim=(-1.2, 1.2), xlabel="x' (mm)",
          ylabel="y' (mm)", title='Doble rendija')
    b.plot(xd*1e3, envolvente, '--', color='0.4', label='Envolvente teórica')
    teorica = envolvente*np.cos(np.pi*d*xd/(lam*z))**2
    b.plot(xd*1e3, teorica, color='#aa4932', lw=1, label='Doble teórica')
    b.plot(eje*1e3, perfiles[2], color='#2454a6', label='FFT2D doble')
    b.set(xlim=(-22, 22), xlabel='x (mm)', ylabel=r'$I(x,0)/I_0$')
    b.legend()
    fig.savefig(OUT/'lab7_punto2_doble.png', dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 3.7), layout='constrained')
    for m, color in [(2, '#2454a6'), (6, '#b53f53')]:
        ax.plot(eje*1e3, perfiles[m], '.-', ms=2.5, color=color, label=f'M = {m}')
    ax.plot(xd*1e3, envolvente, ':', color='0.45', label='Envolvente')
    ax.set(xlim=(-18, 18), xlabel='x (mm)', ylabel=r'$I(x,0)/I_0$')
    ax.legend()
    fig.savefig(OUT/'lab7_punto2_red.png', dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    main()
