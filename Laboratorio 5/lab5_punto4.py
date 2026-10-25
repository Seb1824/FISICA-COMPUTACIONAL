import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = 2e-13
T = 1.
N_PART = 5000
PASOS = np.array([1e-2,5e-3,1e-3,5e-4,1e-4])
DT_FIN = float(PASOS.min())
SEMILLA = 20261025
ALPHA = .3   # 1/s: deriva del control GBM
BETA = .8    # 1/sqrt(s): ruido multiplicativo del control
REPETICIONES = 1000
PARTICULAS_MC = (100,500,1000,5000)


def experimento_acoplado(n_part=N_PART, semilla=SEMILLA):
    rng=np.random.default_rng(semilla)
    n=int(round(T/DT_FIN))
    factores=np.rint(PASOS/DT_FIN).astype(int)
    bloque=int(factores.max())
    if n%bloque or np.any(bloque%factores):
        raise ValueError("La malla fina debe acoplar todos los pasos.")
    w=np.zeros(n_part)
    libre=np.zeros((len(PASOS),n_part))
    gbm=np.ones((len(PASOS),n_part))
    amplitud=np.sqrt(2*D)
    for _ in range(n//bloque):
        dw=np.sqrt(DT_FIN)*rng.standard_normal((bloque,n_part))
        w+=dw.sum(axis=0)
        for j,(dt,factor) in enumerate(zip(PASOS,factores)):
            incrementos=dw.reshape(bloque//factor,factor,n_part).sum(axis=1)
            for incremento in incrementos:
                libre[j]+=amplitud*incremento
                gbm[j]*=1+ALPHA*dt+BETA*incremento
    exacto=amplitud*w
    exacto_gbm=np.exp((ALPHA-BETA**2/2)*T+BETA*w)
    fuertes=np.mean(np.abs(libre-exacto[None,:]),axis=1)
    debiles=np.abs(np.mean(libre**2,axis=1)-2*D*T)
    fuerte_gbm=np.mean(np.abs(gbm-exacto_gbm[None,:]),axis=1)
    momento_continuo=np.exp((2*ALPHA+BETA**2)*T)
    momento_em=np.exp((T/PASOS)*np.log1p((2*ALPHA+BETA**2)*PASOS+ALPHA**2*PASOS**2))
    debil_gbm=np.abs(momento_em-momento_continuo)
    return fuertes,debiles,fuerte_gbm,debil_gbm


def experimento_mc(semilla=SEMILLA+200,repeticiones=REPETICIONES):
    """RMSE de la estimacion de MSD, medido en ensambles independientes."""
    rng=np.random.default_rng(semilla)
    filas=[]
    for n_part in PARTICULAS_MC:
        estimadores=[]
        for inicio in range(0,repeticiones,100):
            cantidad=min(100,repeticiones-inicio)
            z=rng.standard_normal((cantidad,n_part))
            z*=z
            estimadores.extend(2*D*T*z.mean(axis=1))
        errores=np.asarray(estimadores)-2*D*T
        rmse=float(np.sqrt(np.mean(errores**2)))
        teo=2*D*T*np.sqrt(2/n_part)
        filas.append([n_part,rmse,teo,rmse/teo])
    return np.array(filas)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mostrar",action="store_true")
    p.add_argument("--semilla",type=int,default=SEMILLA)
    args=p.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    carpeta=Path(__file__).resolve().parent
    fuerte,debil,control_f,control_d=experimento_acoplado(semilla=args.semilla)
    mc=experimento_mc(semilla=args.semilla+200)
    pf=float(np.polyfit(np.log(PASOS),np.log(control_f),1)[0])
    qd=float(np.polyfit(np.log(PASOS),np.log(control_d),1)[0])
    pmc=float(np.polyfit(np.log(mc[:,0]),np.log(mc[:,1]),1)[0])
    pendiente_muestra=float(np.polyfit(np.log(PASOS),np.log(debil),1)[0])
    limite_redondeo=100*np.finfo(float).eps*np.sqrt(2*D*T)
    filas=[dict(dt_s=float(dt),error_fuerte_m=float(ef),
                 error_debil_muestral_m2=float(ed))
           for dt,ef,ed in zip(PASOS,fuerte,debil)]
    resumen=dict(semilla=args.semilla,n_part=N_PART,D_m2_s=D,
                  error_fuerte_teorico=0.,sesgo_debil_teorico=0.,
                  umbral_redondeo_m=float(limite_redondeo),
                  orden_fuerte_libre=None,orden_debil_libre=None,
                  pendiente_muestral_libre=pendiente_muestra,
                  se_montecarlo_msd_m2=float(2*D*T*np.sqrt(2/N_PART)),
                  control_alpha_s1=ALPHA,control_beta_s_medio=BETA,
                  p_control_fuerte=pf,q_control_debil_analitico=qd,
                  pendiente_montecarlo=pmc,repeticiones_mc=REPETICIONES,datos=filas)
    (carpeta/"lab5_punto4_resumen.json").write_text(json.dumps(resumen,indent=2),encoding="utf-8")
    texto=json.dumps(resumen,indent=2)
    (carpeta/"lab5_punto4_resultados.txt").write_text(texto+"\n",encoding="utf-8")
    print(texto)
    np.savetxt(carpeta/"lab5_punto4_convergencia_libre.csv",
               np.column_stack((PASOS,fuerte,debil)),delimiter=",",comments="",
               header="dt_s,error_fuerte_m,error_debil_muestral_m2")
    np.savetxt(carpeta/"lab5_punto4_control_gbm.csv",
               np.column_stack((PASOS,control_f,control_d)),delimiter=",",comments="",
               header="dt_s,error_fuerte_control,sesgo_debil_analitico_control")
    np.savetxt(carpeta/"lab5_punto4_montecarlo.csv",mc,delimiter=",",comments="",
               header="N_part,RMSE_empirico_m2,RMSE_teorico_m2,razon_empirico_teorico")
    fig,ax=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
    # Los ceros exactos no se sustituyen por errores inventados en escala log.
    positivos=fuerte>0
    ax[0].loglog(PASOS[positivos],fuerte[positivos],"o-",label="Diferencias de redondeo")
    ax[0].axhline(limite_redondeo,color="gray",ls="--",label="Umbral numérico")
    ax[0].set(title="Browniano libre: error fuerte",ylabel="Error [m]")
    ax[0].text(.05,.12,"Error de truncamiento exacto = 0",transform=ax[0].transAxes,fontsize=8)
    ax[1].loglog(PASOS,debil,"o-",label="Error del momento muestral")
    ax[1].axhline(resumen['se_montecarlo_msd_m2'],color="gray",ls="--",label="Escala Monte Carlo")
    ax[1].set(title="Browniano libre: momento segundo",ylabel=r"Error [m$^2$]")
    for a in ax:
        a.set_xlabel(r"Paso $\Delta t$ [s]")
        a.grid(which="both",alpha=.25)
        a.legend(fontsize=7)
    fig.savefig(carpeta/"lab5_punto4_errores_libres.png",dpi=200)
    fig,ax=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
    for a,errores,orden,teorico,titulo in zip(ax,[control_f,control_d],[pf,qd],[.5,1.],
        ["Control GBM: error fuerte","Control GBM: sesgo débil analítico"]):
        a.loglog(PASOS,errores,"o-",label=f"Pendiente medida = {orden:.4f}")
        a.loglog(PASOS,errores[-1]*(PASOS/PASOS[-1])**teorico,"--",label=f"Referencia orden {teorico:g}")
        a.set(title=titulo,xlabel=r"Paso $\Delta t$ [s]",ylabel="Error del control")
        a.grid(which="both",alpha=.25)
        a.legend(fontsize=7)
    fig.savefig(carpeta/"lab5_punto4_control_gbm.png",dpi=200)
    fig,a=plt.subplots(figsize=(8,4.4),layout="constrained")
    a.loglog(mc[:,0],mc[:,1],"o-",label=f"RMSE medido: pendiente {pmc:.4f}")
    a.loglog(mc[:,0],mc[:,2],"k--",label=r"Teoría: $2DT\sqrt{2/N}$")
    a.set(xlabel="Número de partículas N",ylabel=r"RMSE del MSD [m$^2$]",
          title=f"Monte Carlo: {REPETICIONES} ensambles independientes por tamaño")
    a.grid(which="both",alpha=.25)
    a.legend(fontsize=8)
    fig.savefig(carpeta/"lab5_punto4_montecarlo.png",dpi=200)
    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__=="__main__":
    main()
