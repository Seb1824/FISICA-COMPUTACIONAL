import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 4 * np.pi**2
MASAS = np.ones(3)
EPSILON = 0.01
R = 1.0
T_SIM = 2.0
PASOS = (0.01, 0.005, 0.0025, 0.00125)
FACTOR_VELOCIDAD = 1.25


def estado_inicial():
    posiciones = R * np.array([[0.0, 0.0], [1.0, 0.0],
                               [0.5, np.sqrt(3) / 2]])
    centro = np.average(posiciones, axis=0, weights=MASAS)
    relativas = posiciones - centro
    # Valida para el triangulo equilatero de tres masas identicas.
    omega_c = np.sqrt(3 * G * MASAS[0] / (R**2 + EPSILON**2)**1.5)
    velocidades = FACTOR_VELOCIDAD * omega_c * np.column_stack(
        (-relativas[:, 1], relativas[:, 0]))
    return np.r_[posiciones.ravel(), velocidades.ravel()]


def aceleraciones(posiciones, epsilon=EPSILON):
    # desplazamientos[i,j] = r_j - r_i; cada fila corresponde al cuerpo i.
    desplazamientos = posiciones[None, :, :] - posiciones[:, None, :]
    dist2 = np.sum(desplazamientos**2, axis=2) + epsilon**2
    np.fill_diagonal(dist2, np.inf)  # Excluye la autointeraccion j=i.
    return G * np.sum(MASAS[None, :, None] * desplazamientos
                      / dist2[:, :, None]**1.5, axis=1)


def f_sist(u):
    posiciones = u[:6].reshape(3, 2)
    return np.r_[u[6:], aceleraciones(posiciones).ravel()]


def paso_rk4(u, h):
    k1 = f_sist(u)
    k2 = f_sist(u + h * k1 / 2)
    k3 = f_sist(u + h * k2 / 2)
    k4 = f_sist(u + h * k3)
    return u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simular(h):
    if h <= 0:
        raise ValueError("h debe ser positivo.")
    n = int(round(T_SIM / h))
    if not np.isclose(n * h, T_SIM, rtol=0, atol=1e-12):
        raise ValueError("El paso debe dividir el intervalo temporal.")
    t = np.arange(n + 1) * h
    u = np.empty((n + 1, 12))
    u[0] = estado_inicial()
    for i in range(n):
        u[i + 1] = paso_rk4(u[i], h)
    return t, u


def energia(u):
    posiciones = u[:, :6].reshape(-1, 3, 2)
    velocidades = u[:, 6:].reshape(-1, 3, 2)
    cinetica = np.sum(MASAS[None, :, None] * velocidades**2, axis=(1, 2)) / 2
    potencial = np.zeros(len(u))
    for i in range(3):
        for j in range(i + 1, 3):
            distancia = np.sqrt(np.sum((posiciones[:, i] - posiciones[:, j])**2,
                                      axis=1) + EPSILON**2)
            potencial -= G * MASAS[i] * MASAS[j] / distancia
    return cinetica + potencial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    destino = Path(__file__).resolve().parent
    inicial = estado_inicial()
    posiciones = inicial[:6].reshape(3, 2)
    a1_analitica = np.array([6 * np.pi**2, 2 * np.sqrt(3) * np.pi**2])
    a1_numerica = aceleraciones(posiciones, epsilon=0)[0]
    if not np.allclose(a1_analitica, a1_numerica, rtol=1e-13, atol=1e-13):
        raise RuntimeError("La aceleracion inicial no coincide con el calculo analitico.")

    resultados, errores = [], []
    resumen = [
        "Tres cuerpos: RK4, T=2 anos, epsilon=0.01 UA",
        "Energia: masa solar * UA^2 / ano^2",
        f"a1(0), epsilon=0: {a1_analitica} UA/ano^2",
        "Velocidades iniciales [UA/ano]:",
        np.array2string(inicial[6:].reshape(3, 2), precision=9),
        f"Energia inicial E0={energia(inicial[None, :])[0]:.12f}",
    ]
    for h in PASOS:
        t, u = simular(h)
        e = energia(u)
        relativo = np.abs((e - e[0]) / e[0])
        error = float(np.max(relativo))
        errores.append(error)
        resultados.append((h, t, u, relativo))
    ordenes = np.log2(np.array(errores[:-1]) / np.array(errores[1:]))
    with (destino / "punto4_convergencia.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["h_anos", "numero_pasos", "error_relativo_max_energia",
                         "razon_con_paso_anterior", "orden_con_paso_anterior"])
        for j, (h, _, _, _) in enumerate(resultados):
            razon = errores[j - 1] / errores[j] if j else ""
            orden = ordenes[j - 1] if j else ""
            writer.writerow([h, round(T_SIM / h), errores[j], razon, orden])
            resumen.append(f"h={h:.5f}; pasos={round(T_SIM / h)}; "
                           f"error_E={errores[j]:.12e}; razon={razon}; p={orden}")

    h, t, u, _ = resultados[-1]
    r = u[:, :6].reshape(-1, 3, 2)
    v = u[:, 6:].reshape(-1, 3, 2)
    centro = np.average(r, axis=1, weights=MASAS)
    momento = np.sum(MASAS[None, :, None] * v, axis=1)
    distancias = np.column_stack([np.linalg.norm(r[:, i] - r[:, j], axis=1)
                                 for i in range(3) for j in range(i + 1, 3)])
    resumen.extend([
        f"Separacion minima={np.min(distancias):.9f} UA",
        f"Separacion maxima={np.max(distancias):.9f} UA",
        f"Desplazamiento maximo del centro de masa={np.max(np.linalg.norm(centro-centro[0],axis=1)):.3e} UA",
        f"Cambio maximo del momento lineal={np.max(np.linalg.norm(momento-momento[0],axis=1)):.3e}",
    ])
    (destino / "punto4_resultados.txt").write_text("\n".join(resumen) + "\n", encoding="utf-8")
    print("\n".join(resumen))
    np.savetxt(destino / "punto4_trayectoria.csv",
               np.column_stack((t, u, energia(u))), delimiter=",", comments="",
               header="t_anos,x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3,E")

    fig_r, ax_r = plt.subplots(figsize=(7, 6), layout="constrained")
    for i, color in enumerate(("#1565b5", "#c65300", "#279449")):
        ax_r.plot(r[:, i, 0], r[:, i, 1], color=color, label=f"Cuerpo {i+1}")
        ax_r.scatter(r[0, i, 0], r[0, i, 1], color=color, s=30, zorder=4)
    ax_r.scatter(centro[0, 0], centro[0, 1], color="black", marker="x", s=45,
                 label="Centro de masa")
    ax_r.set(xlabel="x [UA]", ylabel="y [UA]",
             title="Tres cuerpos: órbitas acopladas con RK4")
    ax_r.set_aspect("equal", adjustable="box")
    ax_r.grid(alpha=0.3)
    ax_r.legend()
    fig_r.savefig(destino / "punto4_trayectorias.png", dpi=220)

    fig_e, (ax_t, ax_h) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for h, ti, _, error in resultados:
        ax_t.semilogy(ti[1:], np.maximum(error[1:], 1e-16), label=f"h={h:g}")
    ax_t.set(xlabel="Tiempo [años]", ylabel="Error relativo de energía",
             title="Conservación de la energía suavizada")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=8)
    hs = np.array(PASOS)
    ax_h.loglog(hs, errores, "o-", label="Error máximo medido")
    ax_h.loglog(hs, errores[-1] * (hs / hs[-1])**4, "--", label=r"Referencia $C h^4$")
    ax_h.set(xlabel="Paso h [años]", ylabel="Error relativo máximo de energía",
             title="Convergencia hacia orden cuatro")
    ax_h.grid(which="both", alpha=0.3)
    ax_h.legend(fontsize=8)
    fig_e.savefig(destino / "punto4_convergencia.png", dpi=220)
    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
