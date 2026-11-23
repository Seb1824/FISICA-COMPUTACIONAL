"""Orbita ecuatorial efectiva y determinacion subpaso de los perihelios."""
from math import pi, sqrt
import numpy as np
from practica8_rk4 import trayectoria

MU = 4*pi**2
ARCSEC = 180*3600/pi


def orbita(a, e, c, dt, duracion, metodo='rk4', dtype=np.float64):
    rp = a*(1-e)
    vp = sqrt(MU*(1+e)/rp)
    mu = dtype(MU)
    ell = dtype(rp*vp)
    correccion = dtype(3*(float(ell)/c)**2)

    def ecuacion(estado):
        x, y, vx, vy = estado
        r2 = x*x+y*y
        factor = -mu*(dtype(1)+correccion/r2)/(r2*np.sqrt(r2))
        return np.array([vx, vy, factor*x, factor*y], dtype=dtype)

    pasos = int(round(duracion/dt))
    estados = trayectoria(ecuacion, (rp, 0., 0., vp), dt, pasos, metodo, dtype)
    return np.arange(pasos+1)*dt, estados


def hermite(q0, q1, v0, v1, dt, s):
    """Devuelve posicion y derivada respecto a la fraccion de paso s."""
    delta = q1-q0
    b = dt*v0
    c = 3*delta-dt*(2*v0+v1)
    d = -2*delta+dt*(v0+v1)
    return q0+s*(b+s*(c+s*d)), b+s*(2*c+3*s*d)


def perihelios(t, estados, metodo='rk4'):
    estados = estados.astype(np.float64, copy=False)
    q, v = estados[:, :2], estados[:, 2:]
    radial = np.sum(q*v, axis=1)
    cruces = np.flatnonzero((radial[:-1] < 0) & (radial[1:] >= 0))
    # El reposo radial inicial corresponde al primer perihelio.
    tiempos, posiciones = [t[0]], [q[0]]
    for j in cruces:
        dt = t[j+1]-t[j]
        if metodo == 'euler':
            # Interpolar el evento r.v sin imponer derivadas a una recta Euler.
            s = -radial[j]/(radial[j+1]-radial[j])
            tiempos.append(t[j]+s*dt)
            posiciones.append(q[j]+s*(q[j+1]-q[j]))
            continue
        izq, der = 0., 1.
        for _ in range(40):
            s = (izq+der)/2
            pos, tang = hermite(q[j], q[j+1], v[j], v[j+1], dt, s)
            if pos@tang < 0:
                izq = s
            else:
                der = s
        s = (izq+der)/2
        pos, _ = hermite(q[j], q[j+1], v[j], v[j+1], dt, s)
        tiempos.append(t[j]+s*dt)
        posiciones.append(pos)
    return np.array(tiempos), np.array(posiciones)


def avances(posiciones):
    antes, despues = posiciones[:-1], posiciones[1:]
    cruz = antes[:, 0]*despues[:, 1]-antes[:, 1]*despues[:, 0]
    punto = np.sum(antes*despues, axis=1)
    return np.arctan2(cruz, punto)


def precesion_einstein(a, e, c):
    return 6*pi*MU/(a*(1-e**2)*c**2)
