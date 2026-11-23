"""Oscilador relativista en (x,v), control clasico y convergencia."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from practica8_rk4 import trayectoria

OUT = Path(__file__).resolve().parent
U0 = (1.5, 0.0)


def relativista(u):
    x, v = u
    margen = (1-v)*(1+v)
    if margen <= 0:
        raise ValueError('Una etapa RK4 alcanzo la barrera |v|=1.')
    return np.array([v, -x*margen**1.5])


def clasico(u):
    x, v = u
    return np.array([v, -x])


def energia_relativa(estados):
    x, v = estados.T
    energia = 1/np.sqrt((1-v)*(1+v)) + x*x/2
    return (energia-energia[0])/energia[0]


def main():
    h = .01
    t = np.arange(1401)*h
    rel = trayectoria(relativista, U0, h, len(t)-1)
    gal = trayectoria(clasico, U0, h, len(t)-1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    for datos, nombre, color in ((rel, 'Relativista', '#1b5aaa'),
                                 (gal, 'Clásico', '#b84b34')):
        a.plot(t, datos[:, 1], label=nombre, color=color)
        b.plot(datos[:, 0], datos[:, 1], label=nombre, color=color)
    for ax in (a, b):
        for limite in (-1, 1):
            ax.axhline(limite, ls=':', lw=.9, color='0.35')
        ax.set(ylabel=r'$v/c$')
        ax.grid(alpha=.2)
    a.set(xlabel='Tiempo (u. n.)', title='Evolución temporal')
    b.set(xlabel='Posición (u. n.)', title='Espacio de fases')
    a.legend()
    fig.savefig(OUT/'practica8_punto2_oscilador.png', dpi=180)
    plt.close(fig)

    pasos = (.04, .02, .01, .005)
    errores = []
    for dt in pasos:
        estados = trayectoria(relativista, U0, dt, round(20/dt))
        errores.append(np.max(abs(energia_relativa(estados))))
    with (OUT/'practica8_punto2_convergencia.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['h', 'error_maximo_relativo', 'cociente', 'orden'])
        for j, (dt, err) in enumerate(zip(pasos, errores)):
            cociente = err/errores[j+1] if j+1 < len(pasos) else None
            p = np.log2(cociente) if cociente is not None else None
            print(dt, err, cociente, p)
            w.writerow([dt, err, cociente, p])
    print('Rapideces maximas muestreadas:', max(abs(rel[:, 1])),
          max(abs(gal[:, 1])))


if __name__ == '__main__':
    main()
