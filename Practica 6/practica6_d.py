from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
L, VMAX, P = 100, 5, 0.3

def paso(pos, vel, rng):
    # pos se mantiene ordenada circularmente hasta ordenar despues del movimiento.
    hueco = (np.roll(pos, -1)-pos-1) % L
    v = np.minimum(np.minimum(vel+1, VMAX), hueco)
    v = np.maximum(0, v - (rng.random(len(v)) < P))
    pos = (pos+v) % L
    orden = np.argsort(pos)
    return pos[orden], v[orden], v.sum()

def simular(n, pasos, rng, calentamiento=0):
    pos = np.sort(rng.choice(L, n, replace=False))
    vel = np.zeros(n, dtype=int)
    matriz = np.zeros((pasos, L), dtype=int)
    flujos = []
    for t in range(calentamiento+pasos):
        pos, vel, desplazamiento = paso(pos, vel, rng)
        if t >= calentamiento:
            matriz[t-calentamiento, pos] = 1
            flujos.append(desplazamiento/L)
    return matriz, np.mean(flujos)

def main():
    matriz, flujo = simular(25, 200, np.random.default_rng(20261027))
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(matriz, cmap='binary', interpolation='nearest', aspect='auto')
    ax.set(xlabel='Celda de la pista', ylabel='Paso temporal', title='NaSch: L=100, N=25, p=0.3')
    fig.savefig(OUT / 'practica6_d_trafico.png', dpi=180)
    plt.close(fig)
    nvals = np.rint(np.linspace(2, 95, 30)).astype(int)
    rho = nvals/L
    # Repeticiones independientes para reducir variabilidad estadistica.
    resultados = np.array([[simular(n, 2000, np.random.default_rng(10000+100*j+n), 500)[1]
                            for j in range(3)] for n in nvals])
    media, error = resultados.mean(axis=1), resultados.std(axis=1, ddof=1)/np.sqrt(3)
    c = np.argmax(media)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.errorbar(rho, media, yerr=error, marker='o', ms=3, capsize=2, label='NaSch, vmax=5, p=0.3')
    xx = np.linspace(0, 1, 201)
    ax.plot(xx, np.minimum(xx, 1-xx), '--', label='Regla 184 determinista, vmax=1')
    ax.axvline(rho[c], color='gray', ls=':', label=f'Maximo: rho={rho[c]:.2f}')
    ax.set(xlabel='Densidad ρ', ylabel='Flujo J (vehículos/celda/paso)')
    ax.grid(alpha=.3); ax.legend()
    fig.savefig(OUT / 'practica6_d_flujo.png', dpi=180)
    plt.close(fig)
    np.savetxt(OUT / 'practica6_d_flujo.csv', np.column_stack((rho, media, error)), delimiter=',',
               header='densidad,flujo_medio,error_estandar_3_replicas', comments='')
    texto = (f'flujo N=25 y T=200: {flujo:.6f}\n'
             f'densidad de maximo flujo en la malla: {rho[c]:.2f}\n'
             f'flujo maximo: {media[c]:.6f}\n')
    (OUT / 'practica6_d_resultados.txt').write_text(texto, encoding='utf-8')
    print(texto)

if __name__ == '__main__':
    main()
