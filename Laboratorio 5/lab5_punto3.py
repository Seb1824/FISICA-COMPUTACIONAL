import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB = 1.380649e-23
GAMMA = 6*np.pi*1e-3*0.5e-6
MASA = 2e-14
G = 9.81
TEMPERATURAS = np.array([300., 150.])
DIFUSIONES = KB*TEMPERATURAS/GAMMA
ALTURAS = KB*TEMPERATURAS/(MASA*G)
VELOCIDAD = MASA*G/GAMMA
N_PART = 2000
DT = 1e-4
T_MAX = 10.
Y0 = 1e-6
SEMILLA = 20261024


def simular(dt=DT, t_max=T_MAX, n_part=N_PART, semilla=SEMILLA,
            inicio_equilibrio=False, inicio_promedio=1.):
    if dt<=0 or t_max<=0 or n_part<2:
        raise ValueError("Parametros no validos.")
    n = int(round(t_max/dt))
    if not np.isclose(n*dt,t_max):
        raise ValueError("dt debe dividir el intervalo temporal.")
    rng = np.random.default_rng(semilla)
    y = (rng.exponential(ALTURAS[:,None],size=(2,n_part)) if inicio_equilibrio
         else np.full((2,n_part),Y0))
    sigma = np.sqrt(2*DIFUSIONES*dt)[:,None]
    medias = [y.mean(axis=1)]
    tiempos = [0.]
    suma_cola = np.zeros(2)
    cuenta_cola = 0
    reflejos = np.zeros(2,dtype=np.int64)
    for j in range(1,n+1):
        propuesto = y-VELOCIDAD*dt+sigma*rng.standard_normal(y.shape)
        reflejos += (propuesto<0).sum(axis=1)
        y = np.abs(propuesto)
        media = y.mean(axis=1)
        if j*dt>=inicio_promedio:
            suma_cola += media
            cuenta_cola += 1
        if j%100==0 or j==n:
            tiempos.append(j*dt)
            medias.append(media)
    promedio = suma_cola/cuenta_cola if cuenta_cola else np.full(2,np.nan)
    return np.array(tiempos),np.array(medias),y,promedio,reflejos/(n*n_part)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mostrar",action="store_true")
    p.add_argument("--semilla",type=int,default=SEMILLA)
    args=p.parse_args()
    if args.mostrar:
        plt.switch_backend("TkAgg")
    carpeta=Path(__file__).resolve().parent
    t,media,y,promedio,reflejos=simular(semilla=args.semilla)
    h_exp=y.mean(axis=1)
    se=y.std(axis=1,ddof=1)/np.sqrt(N_PART)
    filas=[]
    for i,temp in enumerate(TEMPERATURAS):
        filas.append(dict(T_K=float(temp),D_m2_s=float(DIFUSIONES[i]),
            H_teorica_m=float(ALTURAS[i]),H_exp_m=float(h_exp[i]),
            error_estandar_H_m=float(se[i]),
            error_relativo_H=float(h_exp[i]/ALTURAS[i]-1),
            media_cola_m=float(promedio[i]),fraccion_reflejos=float(reflejos[i])))
    # Control adicional corto: inicia en la ley continua de equilibrio,
    # descarta 0.02 s y promedia hasta 0.2 s para aislar el sesgo de pared.
    controles=[]
    for j,dt in enumerate([DT,DT/2,DT/4]):
        _,_,_,hm,_=simular(dt=dt,t_max=.2,semilla=args.semilla+100+j,
                           inicio_equilibrio=True,inicio_promedio=.02)
        for i,temp in enumerate(TEMPERATURAS):
            controles.append([dt,temp,hm[i],ALTURAS[i],hm[i]/ALTURAS[i]-1])
    resumen=dict(semilla=args.semilla,gamma_kg_s=GAMMA,
                  velocidad_sedimentacion_m_s=VELOCIDAD,y0_m=Y0,datos=filas)
    (carpeta/"lab5_punto3_resumen.json").write_text(json.dumps(resumen,indent=2),encoding="utf-8")
    texto="\n".join(str(f) for f in filas)
    (carpeta/"lab5_punto3_resultados.txt").write_text(texto+"\n",encoding="utf-8")
    print(texto)
    np.savetxt(carpeta/"lab5_punto3_posiciones_finales.csv",y.T,delimiter=",",comments="",
               header="y_300K_m,y_150K_m")
    np.savetxt(carpeta/"lab5_punto3_altura_media.csv",np.column_stack((t,media)),
               delimiter=",",comments="",header="t_s,media_300K_m,media_150K_m")
    np.savetxt(carpeta/"lab5_punto3_refinamiento.csv",controles,delimiter=",",comments="",
               header="dt_s,T_K,H_promedio_m,H_teorica_m,error_relativo")
    fig,ax=plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
    for i,a in enumerate(ax):
        ynm=y[i]*1e9
        limite=max(ynm.max(),6*ALTURAS[i]*1e9)
        yy=np.linspace(0,limite,400)
        ht=ALTURAS[i]*1e9
        he=h_exp[i]*1e9
        a.hist(ynm,bins=35,density=True,alpha=.5,color="#2678ad",label="EM reflejado")
        a.plot(yy,np.exp(-yy/ht)/ht,"k--",lw=1.4,label="Boltzmann continuo")
        a.plot(yy,np.exp(-yy/he)/he,color="#bd5100",lw=1.2,label="Ajuste exponencial MLE")
        a.set(xlabel="Altura y [nm]",ylabel=r"Densidad [nm$^{-1}$]",
              title=f"T = {TEMPERATURAS[i]:g} K")
        a.grid(alpha=.25)
        a.legend(fontsize=7)
    fig.savefig(carpeta/"lab5_punto3_distribuciones.png",dpi=200)
    fig,a=plt.subplots(figsize=(8,4.4),layout="constrained")
    mascara=t>=.1
    for i,color in enumerate(["#1565b5","#c65300"]):
        a.plot(t[mascara],media[mascara,i]*1e9,color=color,lw=.8,label=f"T={TEMPERATURAS[i]:g} K")
        a.axhline(ALTURAS[i]*1e9,color=color,ls="--",label="H teórica")
    a.set(xlim=(.1,T_MAX),xlabel="Tiempo [s]",ylabel="Altura media [nm]",
          title="Altura media después del transitorio inicial")
    a.legend(fontsize=8)
    a.grid(alpha=.25)
    fig.savefig(carpeta/"lab5_punto3_altura_media.png",dpi=200)
    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__=="__main__":
    main()
