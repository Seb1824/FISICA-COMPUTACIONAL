"""Precesion amplificada y resolucion de la correccion relativista."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from practica8_orbitas import avances, orbita, perihelios, precesion_einstein

OUT = Path(__file__).resolve().parent


def main():
    g, sol, c_real, rp = 6.67430e-11, 1.9885e30, 299_792_458., 4.600e10
    rs = 2*g*sol/c_real**2
    epsilon = rs/rp
    print('r_s en m, epsilon:', rs, epsilon)
    print('Spacing sobre 1 float32 / float64:',
          np.spacing(np.float32(1)), np.spacing(np.float64(1)))
    for punto, factor_r in [('perihelio', 1), ('afelio', (1+.2056)/(1-.2056))]:
        delta = 1.5*epsilon*(1+.2056)/factor_r**2
        almacenado = np.float32(1)+np.float32(delta)-np.float32(1)
        print(punto, 'correccion exacta y almacenada:', delta, almacenado)

    a, e, dt = 1., .25, 1e-4
    filas = []
    for c in (100., 150., 200.):
        t, estados = orbita(a, e, c, dt, 3.3)
        tp, qp = perihelios(t, estados)
        giros = avances(qp)
        num, teo = giros.mean(), precesion_einstein(a, e, c)
        filas.append((c, len(giros), num, teo, 100*abs(num/teo-1), num*c*c))
        if c == 100:
            dibujar(t, estados, tp, qp)
    with (OUT/'practica8_punto3_precesion.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['c_UA_anio', 'revoluciones', 'delta_num_rad',
                    'delta_teo_rad', 'error_rel_pct', 'delta_num_c2'])
        w.writerows(filas)
    print(np.array(filas))


def dibujar(t, estados, tp, qp):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    r = estados[:, :2]
    for j in range(len(tp)):
        fin = tp[j+1] if j+1 < len(tp) else t[-1]
        tramo = r[(t >= tp[j]) & (t <= fin)]
        a.plot(*tramo.T, lw=1.05, label=f'Tramo {j+1}')
    a.scatter(0, 0, color='goldenrod', s=35, label='Sol', zorder=4)
    a.set(xlabel='x (UA)', ylabel='y (UA)', aspect='equal')
    a.legend(fontsize=7, loc='lower left')
    angulo = np.r_[0, np.cumsum(avances(qp))]
    b.plot(np.arange(len(qp)), np.degrees(angulo), 'o-', color='#b84b34')
    b.set(xlabel='Revolución', ylabel='Ángulo acumulado (grados)',
          xticks=np.arange(len(qp)))
    b.grid(alpha=.2)
    fig.savefig(OUT/'practica8_punto3_orbita.png', dpi=185)
    plt.close(fig)


if __name__ == '__main__':
    main()
