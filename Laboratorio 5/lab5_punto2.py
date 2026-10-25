import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB = 1.380649e-23
TEMPERATURA = 300.0
GAMMA = 6*np.pi*1e-3*0.5e-6
D = KB*TEMPERATURA/GAMMA
RIGIDECES = np.array([1e-6, 5e-6])
LAMBDA = RIGIDECES/GAMMA
N_PART = 1000
DT = 1e-4
T_MAX = 5.0
X0 = 2e-6
SEMILLA = 20261023


def simular(dt=DT, t_max=T_MAX, n_part=N_PART, semilla=SEMILLA):
    if dt<=0 or n_part<2:
        raise ValueError("El paso debe ser positivo y se necesitan dos particulas.")
    n = int(round(t_max/dt))
    c = 1-LAMBDA*dt
    if not np.isclose(n*dt,t_max) or np.any(np.abs(c)>=1):
        raise ValueError("Paso o parametros no estables/validos.")
    rng = np.random.default_rng(semilla)
    t = np.arange(n+1)*dt
    x = np.full((2, n_part), X0)
    media = np.empty((n+1,2))
    varianza = np.empty((n+1,2))
    media[0] = X0
    varianza[0] = 0
    sigma = np.sqrt(2*D*dt)
    for j in range(n):
        z = rng.standard_normal(n_part)
        x = c[:,None]*x+sigma*z[None,:]
        media[j+1] = x.mean(axis=1)
        varianza[j+1] = x.var(axis=1,ddof=0)
    return t, media, varianza, x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mostrar",action="store_true")
    p.add_argument("--semilla",type=int,default=SEMILLA)
    args = p.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    carpeta = Path(__file__).resolve().parent
    t, media, var, x = simular(semilla=args.semilla)
    continuo_media = X0*np.exp(-t[:,None]*LAMBDA[None,:])
    continuo_var = (D/LAMBDA)[None,:]*(1-np.exp(-2*t[:,None]*LAMBDA[None,:]))
    var_eq = KB*TEMPERATURA/RIGIDECES
    var_em = var_eq/(1-LAMBDA*DT/2)
    cola = t>=1.0
    var_est = var[cola].mean(axis=0)*N_PART/(N_PART-1)
    filas = []
    for i, k in enumerate(RIGIDECES):
        filas.append(dict(k_N_m=float(k), lambda_s1=float(LAMBDA[i]),
            tau_s=float(1/LAMBDA[i]), var_eq_m2=float(var_eq[i]),
            var_EM_estacionaria_m2=float(var_em[i]),
            var_final_m2=float(var[-1,i]),
            var_cola_insesgada_m2=float(var_est[i]),
            media_final_m=float(media[-1,i]),
            sesgo_EM_porcentaje=float(100*(var_em[i]/var_eq[i]-1))))
    # Control adicional del sesgo con dos pasos menores y promedios estacionarios.
    controles = []
    for j, dt in enumerate([DT, DT/2, DT/4]):
        tc, _, vc, _ = simular(dt=dt,t_max=.5,semilla=args.semilla+100+j)
        estimada = vc[tc>=.1].mean(axis=0)*N_PART/(N_PART-1)
        for i,k in enumerate(RIGIDECES):
            controles.append([dt,k,estimada[i],var_eq[i],
                              var_eq[i]/(1-LAMBDA[i]*dt/2)])
    resumen = dict(semilla=args.semilla, gamma_kg_s=GAMMA,D_m2_s=D,
                   n_part=N_PART,datos=filas)
    (carpeta/"lab5_punto2_resumen.json").write_text(json.dumps(resumen,indent=2),encoding="utf-8")
    texto="\n".join(str(f) for f in filas)
    (carpeta/"lab5_punto2_resultados.txt").write_text(texto+"\n",encoding="utf-8")
    print(texto)
    np.savetxt(carpeta/"lab5_punto2_media_varianza.csv",
        np.column_stack((t[::10],media[::10],var[::10],continuo_media[::10],continuo_var[::10])),
        delimiter=",",comments="",
        header="t_s,media_k1_m,media_k2_m,var_k1_m2,var_k2_m2,media_teo_k1_m,media_teo_k2_m,var_teo_k1_m2,var_teo_k2_m2")
    np.savetxt(carpeta/"lab5_punto2_refinamiento.csv",np.array(controles),delimiter=",",comments="",
        header="dt_s,k_N_m,var_estimada_m2,var_eq_continua_m2,var_eq_EM_m2")
    np.savetxt(carpeta/"lab5_punto2_posiciones_finales.csv",x.T,delimiter=",",comments="",
        header="x_k1_m,x_k2_m")
    fig, ejes = plt.subplots(2,2,figsize=(11,7),layout="constrained")
    colores=["#1565b5","#c65300"]
    for i,color in enumerate(colores):
        label=f"k = {RIGIDECES[i]:g} N/m"
        for a in ejes[0]:
            a.plot(t,media[:,i]*1e6,color=color,lw=.7,label=label+" (EM)")
            a.plot(t,continuo_media[:,i]*1e6,"--",color=color,lw=1.3,label="Teoría continua")
            a.set(ylabel=r"Media [$\mu$m]")
        for a in ejes[1]:
            a.plot(t,var[:,i]*1e12,color=color,lw=.6,label=label+" (EM)")
            a.plot(t,continuo_var[:,i]*1e12,"--",color=color,lw=1.3,label="Teoría continua")
            a.axhline(var_em[i]*1e12,color=color,ls=":",lw=1,label="Límite EM")
            a.set(ylabel=r"Varianza [$\mu$m$^2$]")
    for a in ejes[:,0]:
        a.set_xlim(0,.08)
    for a in ejes[:,1]:
        a.set_xlim(0,T_MAX)
    ejes[0,0].set_title("Relajación inicial")
    ejes[0,1].set_title("Intervalo completo")
    for a in ejes.ravel():
        a.set_xlabel("Tiempo [s]")
        a.grid(alpha=.25)
    ejes[0,1].legend(fontsize=7)
    ejes[1,1].legend(fontsize=7)
    fig.savefig(carpeta/"lab5_punto2_media_varianza.png",dpi=200)
    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__=="__main__":
    main()
