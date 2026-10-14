import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


G = 9.81       # m/s^2
L = 1.0        # m
H = 0.01       # s
T_MAX = 15.0   # s
AMPLITUDES = (0.1, 2.0)  # rad; ambos casos parten del reposo


def f_sis(u):
    """u = [theta, omega]; devuelve las pendientes, sin multiplicar por h."""
    theta, omega = u
    return np.array([omega, -(G / L) * np.sin(theta)])


def paso_rk4(u, h):
    k1 = f_sis(u)
    k2 = f_sis(u + (h / 2) * k1)
    k3 = f_sis(u + (h / 2) * k2)
    k4 = f_sis(u + h * k3)
    return u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simular(theta0, h=H, t_max=T_MAX):
    if h <= 0:
        raise ValueError("h debe ser positivo.")
    n = int(round(t_max / h))
    if not np.isclose(n * h, t_max, rtol=0, atol=1e-12):
        raise ValueError("h debe ser positivo y dividir el intervalo temporal.")
    t = np.arange(n + 1, dtype=float) * h
    u = np.empty((n + 1, 2))
    u[0] = [theta0, 0.0]
    for i in range(n):
        u[i + 1] = paso_rk4(u[i], h)
    return t, u


def medir_periodo(t, theta):
    indices = np.flatnonzero((theta[:-1] > 0) & (theta[1:] <= 0))
    if len(indices) < 2:
        raise ValueError("Se necesitan al menos dos cruces descendentes.")
    cruces = t[indices] - theta[indices] * (
        (t[indices + 1] - t[indices])
        / (theta[indices + 1] - theta[indices])
    )
    periodos = np.diff(cruces)
    return float(np.mean(periodos)), cruces


def energia_especifica(u):
    """Energia por unidad de masa, con potencial cero en theta = 0."""
    theta, omega = u[:, 0], u[:, 1]
    return 0.5 * L**2 * omega**2 + G * L * (1 - np.cos(theta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    destino = Path(__file__).resolve().parent
    t0 = 2 * np.pi * np.sqrt(L / G)

    # Comprobacion del calculo manual: theta(0) = pi/3, h = 0.1 s.
    inicial = np.array([np.pi / 3, 0.0])
    k1 = f_sis(inicial)
    k2 = f_sis(inicial + 0.05 * k1)
    resumen = [
        "Pendulo no lineal: RK4, h = 0.01 s, t en [0, 15] s",
        f"k1 (theta0 = pi/3, h = 0.1 s) = {k1}",
        f"k2 (theta0 = pi/3, h = 0.1 s) = {k2}",
        f"Periodo armonico T0 = {t0:.9f} s",
    ]
    filas = []
    resultados = []
    for theta0 in AMPLITUDES:
        t, u = simular(theta0)
        periodo, cruces = medir_periodo(t, u[:, 0])
        energia = energia_especifica(u)
        error_e = float(np.max(np.abs((energia - energia[0]) / energia[0])))
        incremento = 100 * (periodo / t0 - 1)
        filas.append([theta0, periodo, t0, periodo / t0, incremento,
                      len(cruces) - 1, error_e])
        resultados.append((theta0, t, u))
        resumen.append(
            f"theta0 = {theta0:.1f} rad: T = {periodo:.9f} s; "
            f"T/T0 = {periodo / t0:.9f}; aumento = {incremento:.6f} %; "
            f"intervalos = {len(cruces) - 1}; error relativo max. E = {error_e:.3e}"
        )

    with (destino / "punto1_periodos.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["theta0_rad", "T_RK4_s", "T0_s", "T_sobre_T0",
                           "incremento_porcentaje", "periodos_medidos",
                           "error_relativo_max_energia"])
        escritor.writerows(filas)
    (destino / "punto1_resultados.txt").write_text("\n".join(resumen) + "\n", encoding="utf-8")
    print("\n".join(resumen))

    fig_t, ax_t = plt.subplots(figsize=(9, 4.5), layout="constrained")
    fig_f, ejes_f = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    colores = ("#1976b8", "#d35400")
    for (theta0, t, u), color, ax_f in zip(resultados, colores, ejes_f):
        etiqueta = rf"$\theta_0={theta0:.1f}\,\mathrm{{rad}}$"
        ax_t.plot(t, u[:, 0], color=color, label=etiqueta, linewidth=1.5)
        ax_f.plot(u[:, 0], u[:, 1], color=color, linewidth=1.5)
        ax_f.scatter([theta0], [0], color="black", s=22, label="Estado inicial", zorder=3)
        ax_f.set(title=etiqueta, xlabel=r"$\theta$ [rad]", ylabel=r"$\omega$ [rad/s]")
        ax_f.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax_f.grid(alpha=0.3)
        ax_f.legend(fontsize=9)
    ax_t.set(xlabel="Tiempo [s]", ylabel=r"$\theta(t)$ [rad]",
             title="Péndulo no lineal: evolución temporal con RK4", xlim=(0, T_MAX))
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="upper right")
    fig_f.suptitle("Espacio de fases: pequeña y gran amplitud")
    fig_t.savefig(destino / "punto1_theta_t.png", dpi=220)
    fig_f.savefig(destino / "punto1_espacio_fases.png", dpi=220)
    if args.mostrar:
        plt.show()
    plt.close("all")
    print(f"Resultados guardados en: {destino}")


if __name__ == "__main__":
    main()
