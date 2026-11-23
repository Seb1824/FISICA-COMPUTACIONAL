"""Espectro angular con control del periodo FFT y comparacion de Fraunhofer."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from lab7_optica import extension, malla, normalizar

OUT = Path(__file__).resolve().parent
N, LADO, ANCHO, LAM = 1024, 6e-3, 1e-3, 632.8e-9
DX = LADO/N
DISTANCIAS = (0.05, 0.40, 5.0)


def preparar(n):
    x = malla(n, n*DX)
    rendija = abs(x) <= ANCHO/2
    a = (rendija[:, None] & rendija[None, :]).astype(np.float32)
    espectro = np.fft.fft2(np.fft.ifftshift(a))
    f2 = np.fft.fftfreq(n, d=DX)**2
    return x, espectro, f2


def espectro_angular(espectro, f2, z, radio, limitar_ventana):
    n = len(f2)
    propagado = np.empty_like(espectro)
    margen = n*DX/2 - ANCHO/2
    fc = margen/(LAM*np.hypot(z, margen))
    for inicio in range(0, n, 128):
        bloque = slice(inicio, inicio+128)
        radial2 = f2[bloque, None] + f2[None, :]
        s = LAM**2 * radial2
        # k_z-k, evaluado sin restar dos cantidades casi iguales.
        raiz = np.sqrt(np.maximum(1-s, 0))
        fase = -2*np.pi*z/LAM * s/(1+raiz)
        h = np.where(s <= 1, np.exp(1j*fase), 0)
        if limitar_ventana:
            # Coseno suave entre 0.85 fc y fc para reducir el ringing.
            rampa = np.clip((np.sqrt(radial2)/fc-0.85)/0.15, 0, 1)
            h *= 0.5*(1+np.cos(np.pi*rampa))
        propagado[bloque] = espectro[bloque]*h
    u = np.fft.ifft2(propagado)
    mitad = round(radio/DX)
    indices = np.arange(-mitad, mitad+1) % n
    return normalizar(abs(u[np.ix_(indices, indices)])**2)


def campo_lejano(x, z):
    # Evaluacion directa de la suma de Fraunhofer en la malla de salida ASM.
    entrada = malla(N, LADO)
    abertura = entrada[abs(entrada) <= ANCHO/2]
    fase = -2j*np.pi*np.outer(x/(LAM*z), abertura)
    u1 = DX*np.exp(fase).sum(axis=1)
    fuerza = abs(u1)**2
    return normalizar(np.outer(fuerza, fuerza)/(LAM*z)**2)


def nrmse(a, referencia):
    return np.linalg.norm(a-referencia)/np.linalg.norm(referencia)


def main():
    resultados = []
    mapas = {N: {}, 4096: {}}
    # Tres periodos FFT a igual DX: malla pedida y dos ventanas ampliadas.
    for n in (N, 3072, 4096):
        x, espectro, f2 = preparar(n)
        for z in DISTANCIAS:
            radio = 8e-3 if z == 5 and n >= 3072 else 2.8e-3
            m = round(radio/DX)
            eje = x[n//2-m:n//2+m+1]
            limitar = n != N
            intensidad = espectro_angular(espectro, f2, z, radio, limitar)
            referencia = campo_lejano(eje, z)
            interior = abs(eje) <= 2.8e-3
            roi = np.ix_(interior, interior)
            error = nrmse(intensidad[roi], referencia[roi])
            resultados.append((n, n*DX*1000, z, (ANCHO/2)**2/(LAM*z), limitar,
                               error, nrmse(intensidad, referencia)))
            if n in mapas:
                mapas[n][z] = eje, intensidad, referencia
    with (OUT/'lab7_punto4_nrmse.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N_fft', 'ventana_mm', 'z_m', 'N_F', 'filtro_ventana',
                    'NRMSE_ROI_comun', 'NRMSE_mapa'])
        w.writerows(resultados)
    print(np.array(resultados))

    graficar(mapas[N], 'lab7_punto4_base.png')
    graficar(mapas[4096], 'lab7_punto4_ampliado.png')


def graficar(datos, archivo):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), layout='constrained')
    for col, (z, (x, i, ff)) in enumerate(datos.items()):
        eje = x*1e3
        im = axes[0, col].imshow(i, extent=extension(eje), origin='lower',
                                cmap='magma', vmin=0, vmax=1)
        axes[0, col].set(title=f'z = {z:g} m', xlabel='x (mm)', ylabel='y (mm)')
        centro = len(x)//2
        ax = axes[1, col]
        ax.plot(eje, i[centro], label='Espectro angular', color='#285cb2')
        ax.plot(eje, ff[centro], '--', label='Fraunhofer', color='#b53f53')
        ax.set(xlabel='x (mm)', ylabel=r'$I(x,0)/I(0,0)$')
        ax.grid(alpha=.2)
        if col == 2:
            ax.legend(fontsize=8)
    fig.colorbar(im, ax=axes[0], label=r'$I/I(0,0)$', shrink=.8, extend='max')
    fig.savefig(OUT/archivo, dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    main()
