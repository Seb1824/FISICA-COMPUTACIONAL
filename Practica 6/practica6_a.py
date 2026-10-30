from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
X0 = np.array([[1, 1, 0, 1], [1, 1, 1, 0],
               [0, 1, 1, 1], [1, 0, 1, 1]], dtype=int)

def vecinos(x, periodica=False):
    if periodica:
        return sum(np.roll(np.roll(x, di, axis=0), dj, axis=1)
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)
                   if (di, dj) != (0, 0))
    p = np.pad(x, 1)
    return sum(p[1+di:1+di+x.shape[0], 1+dj:1+dj+x.shape[1]]
               for di in (-1, 0, 1) for dj in (-1, 0, 1)
               if (di, dj) != (0, 0))

def paso(x, periodica=False):
    return (vecinos(x, periodica) == 6).astype(int)

def main():
    lineas = []
    for nombre, periodic in [('fija', False), ('periodica', True)]:
        x = X0.copy()
        for t in range(3):
            lineas += [f'{nombre}: t={t}, N={x.sum()}', str(x),
                       f'vecinos:\n{vecinos(x, periodic)}']
            x = paso(x, periodic)
    texto = '\n'.join(lineas) + '\n'
    (OUT / 'practica6_a_resultados.txt').write_text(texto, encoding='utf-8')
    print(texto)

if __name__ == '__main__':
    main()
