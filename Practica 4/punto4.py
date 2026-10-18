import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81
V0 = 25.0
ANGULO = np.deg2rad(40.0)
T = 1.0
PASOS = (0.10, 0.05)


def euler(f, t, u, h):
    return u + h * f(t, u)


def heun(f, t, u, h):
    k1 = f(t, u)
    k2 = f(t + h, u + h * k1)
    return u + h * (k1 + k2) / 2


def rk4(f, t, u, h):
    k1 = f(t, u)
    k2 = f(t + h / 2, u + h * k1 / 2)
    k3 = f(t + h / 2, u + h * k2 / 2)
    k4 = f(t + h, u + h * k3)
    return u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


METODOS = {"Euler": euler, "Heun": heun, "RK4": rk4}


def integrar(f, u0, paso, h, t_final=T):
    if h <= 0:
        raise ValueError("h debe ser positivo.")
    n = int(round(t_final / h))
    if not np.isclose(n * h, t_final, rtol=0, atol=1e-12):
        raise ValueError("h debe dividir el intervalo temporal.")
    t = np.arange(n + 1) * h
    u = np.empty((n + 1, len(u0)))
    u[0] = u0
    for i in range(n):
        u[i + 1] = paso(f, t[i], u[i], h)
    return t, u


def f_proyectil(t, u):
    return np.array([u[2], u[3], 0.0, -G])


def estado_exacto(t):
    t = np.asarray(t)
    vx = V0 * np.cos(ANGULO)
    vy = V0 * np.sin(ANGULO)
    return np.stack((vx * t, vy * t - G * t**2 / 2,
                     np.full_like(t, vx, dtype=float), vy - G * t), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    destino = Path(__file__).resolve().parent
    u0 = estado_exacto(0.0)
    exacto = estado_exacto(T)
    tolerancia = 100 * np.finfo(float).eps * max(np.linalg.norm(exacto[:2]), 1.0)
    resumen = ["Proyectil ideal: T=1 s, v0=25 m/s, angulo=40 grados",
               f"Estado exacto en T: {exacto}",
               f"Umbral para identificar redondeo: {tolerancia:.3e} m"]
    filas, tabla, soluciones = [], [], {}
    for nombre, paso in METODOS.items():
        errores = []
        for h in PASOS:
            t, u = integrar(f_proyectil, u0, paso, h)
            error = float(np.linalg.norm(u[-1, :2] - exacto[:2]))
            errores.append(error)
            soluciones[nombre, h] = (t, u)
            filas.append([nombre, h, *u[-1], error])
        orden = float(np.log2(errores[0] / errores[1])) if min(errores) > tolerancia else None
        tabla.append([nombre, *errores, orden if orden is not None else "N/D: redondeo"])
        resumen.append(f"{nombre}: E(0.1)={errores[0]:.12e} m; "
                       f"E(0.05)={errores[1]:.12e} m; p={orden if orden is not None else 'N/D (redondeo)'}")
    with (destino / "punto4_estados.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "h_s", "x_m", "y_m", "vx_m_s", "vy_m_s", "error_posicion_m"])
        w.writerows(filas)
    with (destino / "punto4_convergencia.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "E_h_0.10_m", "E_h_0.05_m", "orden_empirico"])
        w.writerows(tabla)

    # Control independiente, rotulado de forma separada del enunciado.
    filas_control = []
    resumen.append("CONTROL ADICIONAL: z'=-z, z(0)=1, T=1; solucion exacta exp(-1).")
    for nombre, paso in METODOS.items():
        errores = []
        for h in PASOS:
            _, z = integrar(lambda t, u: -u, np.array([1.0]), paso, h)
            errores.append(float(abs(z[-1, 0] - np.exp(-T))))
        orden = float(np.log2(errores[0] / errores[1]))
        filas_control.append([nombre, *errores, orden])
        resumen.append(f"Control {nombre}: E(0.1)={errores[0]:.12e}; "
                       f"E(0.05)={errores[1]:.12e}; p={orden:.9f}")
    with (destino / "punto4_control_orden.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "error_h_0.10", "error_h_0.05", "orden_empirico"])
        w.writerows(filas_control)
    resumen.append("Heun y RK4 son exactos para aceleracion constante; no se infiere p del redondeo.")
    (destino / "punto4_resultados.txt").write_text("\n".join(resumen) + "\n", encoding="utf-8")
    print("\n".join(resumen))

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    curva = estado_exacto(np.linspace(0, T, 300))
    ax.plot(curva[:, 0], curva[:, 1], color="black", label="Solución exacta", linewidth=1.2)
    for nombre, estilo, color in [("Euler", "s--", "#b44d00"),
                                  ("Heun", "o", "#1671ae"),
                                  ("RK4", "x", "#349641")]:
        _, u = soluciones[nombre, PASOS[0]]
        ax.plot(u[:, 0], u[:, 1], estilo, color=color, label=nombre,
                markersize=5, markerfacecolor="none")
    ax.set(xlabel="x [m]", ylabel="y [m]",
           title="Proyectil ideal: comparación con h = 0.10 s")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(destino / "punto4_trayectorias.png", dpi=220)
    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
