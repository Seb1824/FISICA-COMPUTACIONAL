import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = 1.0          # kg
K = 4.0          # N/m
GAMMA = 0.2      # 1/s; b/m
H = 0.01         # s
T_MAX = 20.0     # s
U0 = np.array([2.0, 0.0])  # x [m], v [m/s]


def f_sist(u, beta):
    x, v = u
    return np.array([v, -GAMMA * v - (K / M) * x - (beta / M) * x**3])


def paso_rk4(u, h, beta):
    k1 = f_sist(u, beta)
    k2 = f_sist(u + h * k1 / 2, beta)
    k3 = f_sist(u + h * k2 / 2, beta)
    k4 = f_sist(u + h * k3, beta)
    return u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simular(beta, h=H, t_max=T_MAX):
    if h <= 0:
        raise ValueError("h debe ser positivo.")
    n = int(round(t_max / h))
    if not np.isclose(n * h, t_max, rtol=0, atol=1e-12):
        raise ValueError("El paso debe dividir el intervalo temporal.")
    t = np.arange(n + 1) * h
    u = np.empty((n + 1, 2))
    u[0] = U0
    for i in range(n):
        u[i + 1] = paso_rk4(u[i], h, beta)
    return t, u


def energia(u, beta):
    x, v = u[:, 0], u[:, 1]
    return M * v**2 / 2 + K * x**2 / 2 + beta * x**4 / 4


def maximos_positivos(t, u):
    """Incluye el maximo inicial y los cruces de v de positivo a negativo.

    Interpola linealmente v para localizar cada maximo. Evalua x mediante
    interpolacion cubica de Hermite usando x y v en los extremos del paso.
    """
    tiempos, amplitudes = [t[0]], [u[0, 0]]
    for i in np.flatnonzero((u[:-1, 1] > 0) & (u[1:, 1] <= 0)):
        s = u[i, 1] / (u[i, 1] - u[i + 1, 1])
        dt = t[i + 1] - t[i]
        x = ((2 * s**3 - 3 * s**2 + 1) * u[i, 0]
             + (s**3 - 2 * s**2 + s) * dt * u[i, 1]
             + (-2 * s**3 + 3 * s**2) * u[i + 1, 0]
             + (s**3 - s**2) * dt * u[i + 1, 1])
        if x > 0:
            tiempos.append(t[i] + s * dt)
            amplitudes.append(x)
    return np.array(tiempos), np.array(amplitudes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    destino = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    fig_a, ax_a = plt.subplots(figsize=(8, 4.5), layout="constrained")
    resumen = ["Duffing amortiguado: RK4, h=0.01 s, t en [0,20] s"]
    filas = []
    for beta, color, nombre in [(1.0, "#c65300", "Duffing"),
                                (0.0, "#1565b5", "Lineal")]:
        t, u = simular(beta)
        e = energia(u, beta)
        tp, ap = maximos_positivos(t, u)
        periodos = np.diff(tp)
        tasas = np.log(ap[:-1] / ap[1:]) / periodos
        etiqueta = rf"{nombre}: $\beta={beta:.0f}$"
        ax.plot(u[:, 0], u[:, 1], color=color, linewidth=1.25, label=etiqueta)
        ax_a.plot(tp, ap, "o-", color=color, label=etiqueta)
        resumen.append(
            f"beta={beta:.1f}: T_primer_ciclo={periodos[0]:.9f} s; "
            f"T_ultimo_ciclo={periodos[-1]:.9f} s; "
            f"alpha_primer_ciclo={tasas[0]:.9f} 1/s; "
            f"alpha_ultimo_ciclo={tasas[-1]:.9f} 1/s; "
            f"ciclos_completos={len(periodos)}; E0={e[0]:.9f} J; "
            f"E20={e[-1]:.9f} J"
        )
        for j, (ti, ai) in enumerate(zip(tp, ap)):
            filas.append([beta, j, ti, ai,
                          periodos[j - 1] if j else "",
                          tasas[j - 1] if j else ""])
        np.savetxt(destino / f"punto2_trayectoria_beta{int(beta)}.csv",
                   np.column_stack((t, u, e)), delimiter=",",
                   header="t_s,x_m,v_m_s,E_J", comments="")
    ax.scatter([2], [0], color="black", s=28, zorder=5, label="Estado inicial")
    ax.scatter([0], [0], color="black", marker="x", s=35, zorder=5, label="Equilibrio")
    ax.set(xlabel="$x$ [m]", ylabel="$v$ [m/s]",
           title="Oscilador de Duffing y oscilador lineal amortiguado")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax_a.set(xlabel="Tiempo [s]", ylabel="Máximos positivos de $x$ [m]",
             title="Disminución de la amplitud: comparación de máximos")
    ax_a.grid(alpha=0.3)
    ax_a.legend()
    fig.savefig(destino / "punto2_espacio_fases.png", dpi=220)
    fig_a.savefig(destino / "punto2_amplitudes.png", dpi=220)
    with (destino / "punto2_maximos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["beta_N_m3", "numero_maximo", "t_s", "amplitud_m",
                         "periodo_anterior_s", "alpha_anterior_1_s"])
        writer.writerows(filas)
    (destino / "punto2_resultados.txt").write_text("\n".join(resumen) + "\n", encoding="utf-8")
    print("\n".join(resumen))
    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
