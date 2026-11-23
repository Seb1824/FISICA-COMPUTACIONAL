"""Oscilador en momento y precesion de Mercurio con controles numericos."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from practica8_orbitas import ARCSEC, avances, orbita, perihelios, precesion_einstein
from practica8_rk4 import trayectoria

OUT = Path(__file__).resolve().parent


def campo(u):
    x, p = u
    return np.array([p/np.hypot(1., p), -x])


def ultrarrelativista():
    dt = .01
    t = np.arange(5001)*dt
    x, p = trayectoria(campo, (10., 0.), dt, len(t)-1).T
    gamma = np.hypot(1., p)
    v = p/gamma
    energia = gamma+x*x/2
    print('max |v|/c:', max(abs(v)),
          'error maximo energia:', max(abs(energia/51.-1)))
    fig, ejes = plt.subplots(2, 1, sharex=True, figsize=(8.5, 4.4),
                             layout='constrained')
    ejes[0].plot(t, x, color='#205bab')
    ejes[1].plot(t, v, color='#a34b34')
    ejes[0].set(ylabel='Posición (u. n.)')
    ejes[1].set(xlabel='Tiempo (u. n.)', ylabel=r'$v/c$')
    for limite in (-1, 1):
        ejes[1].axhline(limite, color='0.45', ls=':', lw=.8)
    for ax in ejes:
        ax.grid(alpha=.2)
    fig.savefig(OUT/'practica8_punto4_oscilador.png', dpi=185)
    plt.close(fig)


def mercurio():
    a, e, c, dt = .3871, .2056, 63241.08, 1e-5
    periodo = a**1.5
    conv = ARCSEC*100/periodo
    teo = precesion_einstein(a, e, c)
    filas = []
    fig, (g1, g2) = plt.subplots(1, 2, figsize=(9.4, 3.5),
                                 layout='constrained')
    prueba = (
        ('RK4 Schwarzschild', c, dt, 'rk4', np.float64),
        ('RK4 Schwarzschild h/2', c, dt/2, 'rk4', np.float64),
        ('RK4 Newton', np.inf, dt, 'rk4', np.float64),
        ('Euler Newton', np.inf, dt, 'euler', np.float64),
        ('RK4 Schwarzschild float32', c, dt, 'rk4', np.float32),
    )
    for j, (etiqueta, luz, paso, metodo, dtype) in enumerate(prueba):
        # Margen de tiempo para obtener diez vueltas completas tambien con Euler.
        t, estados = orbita(a, e, luz, paso, 10.8*periodo, metodo, dtype)
        tp, qp = perihelios(t, estados, metodo)
        if len(tp) < 11:
            raise RuntimeError(f'Faltan perihelios para {etiqueta}.')
        tp, qp = tp[:11], qp[:11]
        giros = avances(qp)
        radios = np.linalg.norm(qp, axis=1)
        filas.append((etiqueta, paso, 10, giros.mean(), conv*giros.mean(),
                      tp[-1], radios[-1]-radios[0]))
        if j in (0, 2, 3):
            g1.plot(range(11), ARCSEC*np.r_[0, np.cumsum(giros)],
                    'o-', ms=3, label=etiqueta)
            g2.plot(range(11), radios-radios[0], 'o-', ms=3)
    g1.set(xlabel='Revolución', ylabel='Avance acumulado (arcsec)')
    g2.set(xlabel='Revolución', ylabel='Cambio en radio de perihelio (UA)')
    g1.legend(fontsize=8)
    for ax in (g1, g2):
        ax.grid(alpha=.2)
    fig.savefig(OUT/'practica8_punto4_mercurio.png', dpi=185)
    plt.close(fig)
    print('Periodo (anios) y teoria (arcsec/siglo):', periodo, teo*conv)
    with (OUT/'practica8_punto4_mercurio.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metodo', 'h_anios', 'vueltas', 'avance_medio_rad',
                    'arcsec_por_siglo_periodo_kepler', 't10_anios', 'cambio_rp_UA'])
        w.writerows(filas)
    for fila in filas:
        print(fila)


if __name__ == '__main__':
    ultrarrelativista()
    mercurio()
