import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB = 1.380649e-23
TEMPERATURA = 300.0
ETA = 1e-3
RADIO = 0.5e-6
GAMMA = 6 * np.pi * ETA * RADIO
D = KB * TEMPERATURA / GAMMA
N_PART = 500
DT = 1e-4
T_MAX = 2.0
SEMILLA = 20261022


def simular(n_part=N_PART, dt=DT, t_max=T_MAX, semilla=SEMILLA):
    if dt <= 0 or n_part <= 0:
        raise ValueError("El paso y el numero de particulas deben ser positivos.")
    n = int(round(t_max / dt))
    if not np.isclose(n * dt, t_max):
        raise ValueError("Parametros de simulacion no validos.")
    rng = np.random.default_rng(semilla)
    t = np.arange(n + 1) * dt
    r = np.zeros((n_part, 2))
    trayectorias = np.zeros((n + 1, min(12, n_part), 2))
    msd = np.zeros(n + 1)
    media = np.zeros((n + 1, 2))
    sigma = np.sqrt(2 * D * dt)
    for j in range(n):
        r += sigma * rng.standard_normal(r.shape)
        trayectorias[j + 1] = r[:trayectorias.shape[1]]
        msd[j + 1] = np.mean(np.sum(r * r, axis=1))
        media[j + 1] = r.mean(axis=0)
    return t, r, trayectorias, msd, media


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mostrar", action="store_true")
    p.add_argument("--semilla", type=int, default=SEMILLA)
    args = p.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    carpeta = Path(__file__).resolve().parent
    t, r, trazas, msd, media = simular(semilla=args.semilla)
    teo = 4 * D * t
    d_exp = msd[-1] / (4 * T_MAX)
    sigma_manual = np.sqrt(2 * D * 0.01)
    resumen = dict(semilla=args.semilla, gamma_kg_s=GAMMA, D_m2_s=D,
                   sigma_manual_m=sigma_manual, x1_manual_m=1.2*sigma_manual,
                   y1_manual_m=-0.8*sigma_manual, msd_final_m2=float(msd[-1]),
                   msd_teorico_final_m2=float(teo[-1]), D_exp_m2_s=float(d_exp),
                   error_relativo_D=float(d_exp/D-1),
                   error_estandar_relativo_D=1/np.sqrt(N_PART))
    (carpeta/"lab5_punto1_resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    texto = "\n".join(f"{k}: {v}" for k, v in resumen.items())
    (carpeta/"lab5_punto1_resultados.txt").write_text(texto+"\n", encoding="utf-8")
    print(texto)
    np.savetxt(carpeta/"lab5_punto1_posiciones_finales.csv", r, delimiter=",",
               header="x_m,y_m", comments="")
    np.savetxt(carpeta/"lab5_punto1_msd.csv",
               np.column_stack((t[::10], msd[::10], teo[::10], media[::10])), delimiter=",",
               header="t_s,MSD_num_m2,MSD_teo_m2,media_x_m,media_y_m", comments="")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for j in range(trazas.shape[1]):
        ax[0].plot(trazas[:, j, 0]*1e6, trazas[:, j, 1]*1e6, lw=0.65)
    ax[0].scatter([0], [0], s=22, color="black", zorder=4)
    ax[0].set_title("12 trayectorias de 500 partículas")
    ax[1].scatter(r[:, 0]*1e6, r[:, 1]*1e6, s=9, alpha=.55, color="#176dab")
    ax[1].set_title("Distribución espacial en t = 2 s")
    for a in ax:
        a.set(xlabel=r"$x$ [$\mu$m]", ylabel=r"$y$ [$\mu$m]")
        a.set_aspect("equal", adjustable="box")
        a.grid(alpha=.25)
    fig.savefig(carpeta/"lab5_punto1_trayectorias.png", dpi=200)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    sigma_um = np.sqrt(2 * D * T_MAX)*1e6
    xx = np.linspace(-4*sigma_um, 4*sigma_um, 400)
    gauss = np.exp(-xx**2/(2*sigma_um**2))/(np.sqrt(2*np.pi)*sigma_um)
    for j, a in enumerate(ax):
        a.hist(r[:, j]*1e6, bins=25, density=True, alpha=.55, label="Ensamble")
        a.plot(xx, gauss, "k--", label="Gaussiana teórica")
        a.set(xlabel=("x" if j==0 else "y")+r" [$\mu$m]",
              ylabel=r"Densidad [$\mu$m$^{-1}$]")
        a.legend(fontsize=8)
        a.grid(alpha=.2)
    fig.savefig(carpeta/"lab5_punto1_distribucion.png", dpi=200)
    fig, a = plt.subplots(figsize=(8, 4.4), layout="constrained")
    a.fill_between(t, teo*(1-1.96/np.sqrt(N_PART))*1e12,
                   teo*(1+1.96/np.sqrt(N_PART))*1e12, alpha=.2,
                   label="Banda puntual aproximada del 95 %")
    a.plot(t, msd*1e12, label="MSD numérico", lw=1)
    a.plot(t, teo*1e12, "k--", label=r"$4Dt$")
    a.set(xlabel="Tiempo [s]", ylabel=r"MSD [$\mu$m$^2$]", title="Difusión libre 2D")
    a.legend(fontsize=8)
    a.grid(alpha=.25)
    fig.savefig(carpeta/"lab5_punto1_msd.png", dpi=200)
    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
