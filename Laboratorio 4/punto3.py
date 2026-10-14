import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

W0_CUADRADO = 9.81    # 1/s^2
GAMMA = 0.2          # 1/s
L = 1.0              # m
F0 = 0.5             # N/kg: aceleracion tangencial de forzamiento
Q0 = F0 / L          # 1/s^2: amplitud angular de forzamiento
WD = 0.667           # rad/s
H = 0.01             # s
T_MAX = 60.0         # s: intervalo solicitado
T_CONTROL = 180.0    # s: verificacion adicional de regimen permanente
T_FORZAMIENTO = 2 * np.pi / WD


def f_sist(t, u):
    theta, omega = u
    return np.array([omega, -GAMMA * omega - W0_CUADRADO * np.sin(theta)
                     + Q0 * np.cos(WD * t)])


def paso_rk4(t, u, h):
    k1 = f_sist(t, u)
    k2 = f_sist(t + h / 2, u + h * k1 / 2)
    k3 = f_sist(t + h / 2, u + h * k2 / 2)
    k4 = f_sist(t + h, u + h * k3)
    return u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simular(t_max=T_MAX, h=H):
    if h <= 0:
        raise ValueError("h debe ser positivo.")
    n = int(round(t_max / h))
    if not np.isclose(n * h, t_max, rtol=0, atol=1e-12):
        raise ValueError("El paso debe dividir el intervalo temporal.")
    t = np.arange(n + 1, dtype=float) * h
    u = np.zeros((n + 1, 2))
    for i in range(n):
        u[i + 1] = paso_rk4(t[i], u[i], h)
    return t, u


def interpolar_estado(t_consulta, t, u):
    return np.column_stack([np.interp(t_consulta, t, u[:, j]) for j in (0, 1)])


def energia_especifica(u):
    """Energia mecanica por unidad de masa, en J/kg."""
    theta, omega = u[:, 0], u[:, 1]
    return L**2 * (omega**2 / 2 + W0_CUADRADO * (1 - np.cos(theta)))


def integrar_trapecio(t, y):
    return float(np.sum((y[1:] + y[:-1]) * np.diff(t) / 2))


def ciclo_final(t, u):
    n = int(np.floor(t[-1] / T_FORZAMIENTO))
    if n < 1:
        raise ValueError("El intervalo debe contener un ciclo completo.")
    a, b = (n - 1) * T_FORZAMIENTO, n * T_FORZAMIENTO
    tc = np.r_[a, t[(t > a) & (t < b)], b]
    uc = interpolar_estado(tc, t, u)
    omega = uc[:, 1]
    perdida = integrar_trapecio(tc, L**2 * GAMMA * omega**2)
    entrada = integrar_trapecio(tc, L**2 * Q0 * omega * np.cos(WD * tc))
    e = energia_especifica(uc)
    cambio = float(e[-1] - e[0])
    return tc, uc, perdida, entrada, cambio


def comparar_periodicidad(t, u, tc, uc):
    previo = interpolar_estado(tc - T_FORZAMIENTO, t, u)
    diferencia = uc - previo
    norma = np.hypot(diferencia[:, 0], diferencia[:, 1] / np.sqrt(W0_CUADRADO))
    return float(np.max(norma) / np.max(np.abs(uc[:, 0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    destino = Path(__file__).resolve().parent

    t, u = simular()
    tr, ur = simular(T_CONTROL)
    ciclos = [("Solicitada", t, u), ("Control", tr, ur)]
    datos = []
    resumen = [
        "Pendulo forzado y amortiguado: RK4 con h=0.01 s",
        "L=1 m; energias por unidad de masa (J/kg).",
        f"Periodo de forzamiento Td={T_FORZAMIENTO:.9f} s",
    ]
    for nombre, ti, ui in ciclos:
        tc, uc, perdida, entrada, cambio = ciclo_final(ti, ui)
        periodicidad = comparar_periodicidad(ti, ui, tc, uc)
        datos.append([nombre, tc[0], tc[-1], perdida, entrada, cambio,
                      cambio - (entrada - perdida), periodicidad])
        resumen.append(
            f"{nombre}: ciclo [{tc[0]:.9f},{tc[-1]:.9f}] s; "
            f"D={perdida:.12e} J/kg; W={entrada:.12e} J/kg; "
            f"Delta_e={cambio:.12e} J/kg; "
            f"residuo_balance={cambio - (entrada - perdida):.3e} J/kg; "
            f"diferencia_relativa_entre_ciclos={periodicidad:.6e}"
        )

    tc, uc, perdida, entrada, cambio = ciclo_final(tr, ur)
    consulta = tc[0] + np.mod(t, T_FORZAMIENTO)
    referencia = interpolar_estado(consulta, tr, ur)
    amplitud = float(np.max(np.abs(uc[:, 0])))
    error = np.hypot(u[:, 0] - referencia[:, 0],
                     (u[:, 1] - referencia[:, 1]) / np.sqrt(W0_CUADRADO)) / amplitud
    # Primer instante tras el cual el error permanece bajo 1% hasta t=60 s.
    error_futuro = np.maximum.accumulate(error[::-1])[::-1]
    candidatos = np.flatnonzero(error_futuro <= 0.01)
    asentamiento = float(t[candidatos[0]]) if len(candidatos) else None
    resumen.extend([
        f"Amplitud angular de referencia={amplitud:.9f} rad",
        f"Tiempo de asentamiento al 1% (criterio indicado)={asentamiento} s",
        f"Error relativo del estado a t=60 s={error[-1]:.9e}",
        f"Potencia media disipada permanente={perdida / T_FORZAMIENTO:.12e} W/kg",
    ])
    (destino / "punto3_resultados.txt").write_text("\n".join(resumen) + "\n", encoding="utf-8")
    print("\n".join(resumen))
    with (destino / "punto3_ciclos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["corrida", "inicio_s", "fin_s", "D_J_kg", "W_J_kg",
                         "Delta_e_J_kg", "residuo_balance_J_kg", "error_periodicidad"])
        writer.writerows(datos)
    np.savetxt(destino / "punto3_trayectoria.csv",
               np.column_stack((t, u, energia_especifica(u), error)), delimiter=",",
               header="t_s,theta_rad,omega_rad_s,e_J_kg,error_relativo_estado", comments="")

    fig_t, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    ax.plot(t, u[:, 0], color="#1565b5", linewidth=1.2, label="RK4: desde el reposo")
    ax.plot(t, referencia[:, 0], "--", color="#b55000", linewidth=1,
            label="Respuesta periódica de referencia")
    if asentamiento is not None:
        ax.axvspan(asentamiento, T_MAX, alpha=0.1, color="green",
                   label="Error del estado menor al 1 %")
    ax.set(xlabel="Tiempo [s]", ylabel=r"$\theta(t)$ [rad]", xlim=(0, T_MAX),
           title="Péndulo forzado y amortiguado: transitorio y respuesta periódica")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig_t.savefig(destino / "punto3_theta_t.png", dpi=220)

    fig_f, ejes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ultimo = t >= T_MAX - T_FORZAMIENTO
    ejes[0].plot(u[:, 0], u[:, 1], color="#8d9aa8", linewidth=0.8,
                 label="Trayectoria: 0 a 60 s")
    ejes[0].scatter([0], [0], color="black", s=22, zorder=4, label="Estado inicial")
    for eje in ejes:
        eje.plot(u[ultimo, 0], u[ultimo, 1], color="#c65300", linewidth=1.2,
                 label="Último período: antes de 60 s")
        eje.plot(uc[:, 0], uc[:, 1], "--", color="#1565b5", linewidth=1.3,
                 label="Órbita periódica de control")
        eje.set(xlabel=r"$\theta$ [rad]", ylabel=r"$\omega$ [rad/s]")
        eje.grid(alpha=0.3)
        eje.xaxis.set_major_locator(plt.MaxNLocator(5))
        eje.legend(fontsize=7)
    ejes[0].set_title("Evolución completa")
    ejes[1].set_title("Aproximación al régimen permanente")
    fig_f.savefig(destino / "punto3_espacio_fases.png", dpi=220)
    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
