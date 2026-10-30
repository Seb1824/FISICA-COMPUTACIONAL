from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ALPHA, DX = 0.02, 0.1

def simular(dt):
    x = np.linspace(0, 1, 11)
    # Perfil triangular no negativo con perturbacion de alta frecuencia.
    u = 100*np.minimum(x, 1-x) + 0.5*np.sin(9*np.pi*x)
    estados = [u.copy()]
    r = ALPHA*dt/DX**2
    # Hasta 5 s para dt=0.2; para dt=0.3 el ultimo punto previo es 4.8 s.
    for _ in range(int(np.floor(5.0/dt + 1e-10))):
        u = u.copy()
        u[1:-1] = estados[-1][1:-1] + r*(estados[-1][:-2]-2*estados[-1][1:-1]+estados[-1][2:])
        u[0] = u[-1] = 0
        estados.append(u)
    return x, np.array(estados), r

def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    lines = []
    for ax, dt in zip(axes, (0.2, 0.3)):
        x, e, r = simular(dt)
        for t in (0, 1.2, 2.4, 3.6, 4.8):
            ax.plot(x, e[round(t/dt)], marker='.', label=f'{t:g} s')
        ax.set(title=f'dt={dt:g} s; r={r:.1f}', xlabel='x (m)', ylabel='Temperatura (°C)')
        ax.grid(alpha=.3)
        ax.legend()
        fin = e[round(4.8/dt)]
        lines.append(f'dt={dt}, r={r:.2f}, t=4.8, min={fin.min():.6f}, max={fin.max():.6f}, centro={fin[5]:.6f}, ultimo_t={(len(e)-1)*dt:g}, ultimo_max={e[-1].max():.6f}')
    fig.savefig(OUT / 'practica6_c_difusion.png', dpi=180)
    plt.close(fig)
    (OUT / 'practica6_c_resultados.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
