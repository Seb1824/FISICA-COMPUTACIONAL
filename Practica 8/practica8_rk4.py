"""Integracion RK4 clasica para sistemas autonomos."""
import numpy as np


def paso_rk4(f, u, h):
    k1 = f(u)
    k2 = f(u + (h/2)*k1)
    k3 = f(u + (h/2)*k2)
    k4 = f(u + h*k3)
    return u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def trayectoria(f, u0, h, pasos, metodo='rk4', dtype=np.float64):
    historia = np.empty((pasos+1, len(u0)), dtype=dtype)
    historia[0] = u = np.array(u0, dtype=dtype)
    h = dtype(h)
    for i in range(1, pasos+1):
        if metodo == 'rk4':
            u = paso_rk4(f, u, h)
        elif metodo == 'euler':
            u = u + h*f(u)
        else:
            raise ValueError(f'Metodo desconocido: {metodo}')
        historia[i] = u
    return historia
