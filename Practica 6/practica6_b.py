from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
X0 = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0], dtype=int)

def paso(x, regla):
    izq, der = np.roll(x, 1), np.roll(x, -1)
    codigo = 4*izq + 2*x + der
    return ((regla >> codigo) & 1).astype(int)

def evolucion(x, regla, pasos):
    estados = [x.copy()]
    for _ in range(pasos):
        estados.append(paso(estados[-1], regla))
    return np.array(estados)

def main():
    lineas = []
    for regla in (90, 184):
        e = evolucion(X0, regla, 50)
        lineas.extend([f'Regla {regla}:', *(f't={t}: {e[t].tolist()} N={e[t].sum()}' for t in range(3)),
                       f'N en t=0..50: {e.sum(axis=1).tolist()}'])
    (OUT / 'practica6_b_resultados.txt').write_text('\n'.join(lineas)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, regla in zip(axes, (90, 184)):
        x = np.zeros(100, dtype=int)
        if regla == 90:
            x[50] = 1
        else:
            x = np.random.default_rng(20261027).binomial(1, 0.4, 100)
        e = evolucion(x, regla, 50)
        ax.imshow(e, interpolation='nearest', cmap='binary', aspect='auto', origin='upper')
        ax.set(title=f'Regla {regla}; N inicial = {x.sum()}', xlabel='Celda', ylabel='Paso temporal')
    fig.savefig(OUT / 'practica6_b_espaciotiempo.png', dpi=180)
    plt.close(fig)
    print('\n'.join(lineas[:8]))

if __name__ == '__main__':
    main()
