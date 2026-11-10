"""Tablas livianas para el libro, derivadas de los checkpoints crudos (que no se versionan por su tamaño).

No ajusta ni optimiza nada nuevo, salvo reconstruir los pesos de RegionalMAPEM de la repetición 0 para conocer la
región de cada cliente de evaluación; la reconstrucción se verifica contra las predicciones guardadas.
Salida: experiments/resultados/tablas/libro/*.csv

Uso: python experiments/confirmatorio/derivar_tablas_libro.py
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import *  # noqa: E402,F403
from correr_anidado import ORIGINAL, mapem, ruta  # noqa: E402
from robustez import ESCENARIOS, ruta_esc  # noqa: E402

SAL = RESULTADOS / "tablas" / "libro"
SAL.mkdir(parents=True, exist_ok=True)
COMB = ["mapem_sin_prior", "mapem_K1", "mapem_regiones_duras", "promedio_simple", "stacking_logistico"]


def pliegue(ds, r, f):
    d = {m: pickle.load(open(ruta(ds, r, f, m), "rb")) for m in BENCHMARKS}
    d.update(pickle.load(open(ruta(ds, r, f, "_combinadores"), "rb")))
    return d


def calibracion(mejor):
    """Curva de calibración de Telco con las predicciones del bucle externo de las 5 repeticiones (10 bins)."""
    _, y = cargar("telco")
    acum = {m: ([], []) for m in BENCHMARKS + [ORIGINAL] + COMB}
    for r in range(5):
        for f, (_, te) in enumerate(pliegues_externos(y, r)):
            d = pliegue("telco", r, f)
            for m in acum:
                acum[m][0].append(d[m]["pte"]); acum[m][1].append(y[te])
    filas = []
    for m, (P, Y) in acum.items():
        p, yy = np.concatenate(P), np.concatenate(Y)
        b = np.minimum((p * 10).astype(int), 9)
        for k in range(10):
            s = b == k
            if s.any():
                filas.append({"modelo": m, "bin": k, "n": int(s.sum()), "p_media": float(p[s].mean()), "frecuencia_observada": float(yy[s].mean())})
    pd.DataFrame(filas).to_csv(SAL / "calibracion_telco.csv", index=False)
    print("calibración lista; mejor benchmark:", mejor, flush=True)


def errores(mejor):
    """Log-loss por subgrupo (contrato, tenure y región) en los 5 pliegues de la repetición 0."""
    X, y = cargar("telco")
    filas, verif = [], []
    for f, (tr, te) in enumerate(pliegues_externos(y, 0)):
        d = pliegue("telco", 0, f)
        O = np.column_stack([d[m]["oof"] for m in BENCHMARKS])
        T = np.column_stack([d[m]["pte"] for m in BENCHMARKS])
        pp = preprocesador("telco").fit(X.iloc[tr])
        Za, Zt = np.asarray(pp.transform(X.iloc[tr]), float), np.asarray(pp.transform(X.iloc[te]), float)
        bp = d[ORIGINAL]["params"]
        mo = mapem(Za, y[tr], O, seed_for("model", 0, f, 100), bp["K"], bp["c"], bp["covarianza"])
        rec = mo.predict_proba_from_predictions(Zt, T)[:, 1]
        verif.append(float(np.max(np.abs(rec - d[ORIGINAL]["pte"]))))
        region = mo._regions(Zt).argmax(1)
        ten = pd.to_numeric(X.iloc[te]["tenure"]).to_numpy()
        grupos = {"contrato": X.iloc[te]["Contract"].to_numpy(),
                  "tenure": pd.cut(ten, [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str),
                  "region": np.array([f"región {k}" for k in region])}
        yt = y[te]
        for nombre in (ORIGINAL, mejor, "mapem_K1"):
            p = np.clip(d[nombre]["pte"], EPS, 1 - EPS)
            ll = -(yt * np.log(p) + (1 - yt) * np.log(1 - p))
            for g, vals in grupos.items():
                for v in np.unique(vals):
                    s = vals == v
                    filas.append({"pliegue": f, "grupo": g, "valor": v, "modelo": nombre, "n": int(s.sum()),
                                  "churn": float(yt[s].mean()), "log_loss": float(ll[s].mean()), "K": bp["K"]})
    pd.DataFrame(filas).to_csv(SAL / "errores_subgrupos.csv", index=False)
    print("errores listos; máxima diferencia al reconstruir RegionalMAPEM:", max(verif), flush=True)
    return max(verif)


def robustez():
    filas, conf, seg = [], [], []
    for e in ESCENARIOS:
        for f in range(5):
            s = pickle.load(open(ruta_esc(e, f, "_resumen"), "rb"))
            filas += s["filas"]
            conf += [{**c, "escenario": e} for c in s.get("conforme", [])]
            seg += [{"escenario": e, "pliegue": f, "modelo": m, "seg": v} for m, v in s["seg"].items()]
    pd.DataFrame(filas).to_csv(SAL / "robustez_resumen.csv", index=False)
    pd.DataFrame(conf).to_csv(SAL / "conforme.csv", index=False)
    pd.DataFrame(seg).to_csv(SAL / "robustez_tiempos.csv", index=False)
    print("robustez lista", flush=True)


def finales():
    """Métricas de cada modelo en las dos corridas únicas: cambio de distribución por tenure y hold-out."""
    filas = []
    for cual, (r, f) in {"tenure": (8, 0), "holdout": (9, 0)}.items():
        X, y = cargar("telco")
        if cual == "tenure":
            ten = pd.to_numeric(X["tenure"]).to_numpy()
            yte = y[ten > 24]
        else:
            yte = cargar_holdout()[1]
        d = pliegue(f"telco_{cual}", r, f)
        for m in BENCHMARKS + [ORIGINAL] + COMB:
            filas.append({"corrida": cual, "modelo": m, "n": len(yte), "churn": float(yte.mean()), **metricas(yte, d[m]["pte"], d[m]["umbral"])})
    pd.DataFrame(filas).to_csv(SAL / "corridas_finales.csv", index=False)
    print("corridas finales listas", flush=True)


def conjuntos():
    """Tamaño, dimensión antes y después del preprocesamiento, desbalance y faltantes de cada conjunto."""
    filas = []
    for ds in DATASETS:
        X, y = cargar(ds)
        Z = preprocesador(ds).fit_transform(X, y)
        filas.append({"conjunto": ds, "n": len(y), "p_original": X.shape[1], "p_preprocesado": Z.shape[1],
                      "churn": float(y.mean()), "celdas_faltantes": float(X.isna().mean().mean()),
                      "repeticiones": REPETICIONES[ds], "pliegues_externos": 5 * REPETICIONES[ds]})
    pd.DataFrame(filas).to_csv(SAL / "conjuntos.csv", index=False)
    print("conjuntos listos", flush=True)


if __name__ == "__main__":
    import json
    if sys.argv[1:] == ["conjuntos"]:
        conjuntos()
        sys.exit()
    mejor = json.loads((RESULTADOS / "tablas" / "veredicto_R3.json").read_text(encoding="utf-8"))["mejor_benchmark"]["telco"]
    calibracion(mejor)
    errores(mejor)
    robustez()
    finales()
