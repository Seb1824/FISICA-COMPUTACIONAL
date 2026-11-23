"""Malla centrada y propagacion de Fraunhofer compartidas por los ejercicios."""
import numpy as np


def malla(n, lado):
    return (np.arange(n) - n // 2) * (lado / n)


def fraunhofer(apertura, dx, lam, z):
    n = apertura.shape[0]
    frecuencia = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    pantalla = lam * z * frecuencia
    espectro = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(apertura)))
    # Se omiten las fases de modulo uno: no cambian intensidad ni energia.
    return pantalla, espectro * (dx**2 / (lam * z))


def normalizar(intensidad):
    return intensidad / intensidad[intensidad.shape[0] // 2,
                                   intensidad.shape[1] // 2]


def extension(eje):
    paso = eje[1] - eje[0]
    return [eje[0] - paso/2, eje[-1] + paso/2] * 2
