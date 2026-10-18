import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU = 3.986e14        # GM [m^3/s^2]
R0 = 7.0e6          # m
V0 = np.sqrt(MU / R0)
OMEGA = np.sqrt(MU / R0**3)
PERIODO = 2 * np.pi / OMEGA
PASOS = (20.0, 10.0)
T_SIM = int(np.floor(PERIODO / max(PASOS))) * max(PASOS)
U0 = np.array([R0, 0.0, 0.0, V0])


def aceleracion(r):
    radio = np.linalg.norm(r)
    if radio == 0:
        raise ValueError("La aceleracion es singular en el origen.")
    return -MU * r / radio**3


def f_sist(u):
    return np.r_[u[2:], aceleracion(u[:2])]


def euler_cromer(u, h):
    v_nueva = u[2:] + h * aceleracion(u[:2])
    r_nueva = u[:2] + h * v_nueva
    return np.r_[r_nueva, v_nueva]


def rk4(u, h):
    k1 = f_sist(u)
    k2 = f_sist(u + h * k1 / 2)
    k3 = f_sist(u + h * k2 / 2)
    k4 = f_sist(u + h * k3)
    return u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


METODOS = {"Euler-Cromer": (euler_cromer, 1), "RK4": (rk4, 4)}


def simular(nombre, h):
    paso, evaluaciones_por_paso = METODOS[nombre]
    if h <= 0:
        raise ValueError("h debe ser positivo.")
    n = int(round(T_SIM / h))
    if not np.isclose(n * h, T_SIM, rtol=0, atol=1e-10):
        raise ValueError("h debe dividir el intervalo temporal comun.")
    t = np.arange(n + 1) * h
    u = np.empty((n + 1, 4))
    u[0] = U0
    for i in range(n):
        u[i + 1] = paso(u[i], h)
    return t, u, n * evaluaciones_por_paso


def energia_especifica(u):
    return np.sum(u[:, 2:]**2, axis=1) / 2 - MU / np.linalg.norm(u[:, :2], axis=1)


def solucion_exacta(t):
    fase = OMEGA * np.asarray(t)
    return np.stack((R0 * np.cos(fase), R0 * np.sin(fase),
                     -V0 * np.sin(fase), V0 * np.cos(fase)), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    destino = Path(__file__).resolve().parent
    resumen = [f"v0={V0:.12f} m/s", f"Periodo teorico={PERIODO:.12f} s",
               f"Intervalo comun=[0,{T_SIM:.0f}] s; diferencia con T={PERIODO-T_SIM:.9f} s",
               f"Energia especifica inicial={-MU/(2*R0):.12f} J/kg"]
    resultados = {}
    filas = []
    for nombre in METODOS:
        for h in PASOS:
            t, u, costo = simular(nombre, h)
            e = energia_especifica(u)
            relativo = np.abs((e - e[0]) / e[0])
            delta_e = float(np.max(relativo))
            delta_r = float(np.max(np.abs(np.linalg.norm(u[:, :2], axis=1) / R0 - 1)))
            exacta = solucion_exacta(t)
            error_posicion = float(np.max(np.linalg.norm(u[:, :2] - exacta[:, :2], axis=1)))
            filas.append([nombre, h, len(t) - 1, costo, delta_e, delta_r, error_posicion])
            resultados[nombre, h] = (t, u, relativo)
            resumen.append(f"{nombre}, h={h:g} s: pasos={len(t)-1}; evaluaciones={costo}; "
                           f"delta_E={delta_e:.12e}; delta_r={delta_r:.12e}; "
                           f"error_posicion_max={error_posicion:.9f} m")
            nombre_archivo = "ec" if nombre == "Euler-Cromer" else "rk4"
            np.savetxt(destino / f"punto5_{nombre_archivo}_h{int(h)}.csv",
                       np.column_stack((t, u, e, relativo)), delimiter=",", comments="",
                       header="t_s,x_m,y_m,vx_m_s,vy_m_s,energia_J_kg,error_relativo_E")
    with (destino / "punto5_comparacion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "h_s", "pasos", "evaluaciones_dinamica",
                    "delta_E", "delta_r", "error_posicion_max_m"])
        w.writerows(filas)
    (destino / "punto5_resultados.txt").write_text("\n".join(resumen) + "\n", encoding="utf-8")
    print("\n".join(resumen))

    fig_r, ejes_r = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    fig_e, ejes_e = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    curva = solucion_exacta(np.linspace(0, PERIODO, 600))
    for h, ax_r, ax_e in zip(PASOS, ejes_r, ejes_e):
        ax_r.plot(curva[:, 0] / 1e6, curva[:, 1] / 1e6, "k--", linewidth=1,
                  label="Órbita circular exacta")
        for nombre, color in [("Euler-Cromer", "#1565b5"), ("RK4", "#c65300")]:
            t, u, error = resultados[nombre, h]
            ax_r.plot(u[:, 0] / 1e6, u[:, 1] / 1e6, color=color, linewidth=1.1, label=nombre)
            ax_e.semilogy(t[1:], np.maximum(error[1:], 1e-18), color=color, label=nombre)
        ax_r.scatter([0], [0], color="black", s=22)
        ax_r.set(xlabel=r"$x$ [$10^6$ m]", ylabel=r"$y$ [$10^6$ m]", title=f"h = {h:g} s")
        ax_r.set_aspect("equal", adjustable="box")
        ax_r.grid(alpha=0.3)
        ax_r.legend(fontsize=7)
        ax_e.set(xlabel="Tiempo [s]", ylabel="Error relativo de energía", title=f"h = {h:g} s")
        ax_e.grid(alpha=0.3)
        ax_e.legend(fontsize=8)
    fig_r.suptitle("Órbita gravitacional: comparación de integradores")
    fig_e.suptitle("Conservación de la energía mecánica específica")
    fig_r.savefig(destino / "punto5_trayectorias.png", dpi=220)
    fig_e.savefig(destino / "punto5_error_energia.png", dpi=220)
    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
