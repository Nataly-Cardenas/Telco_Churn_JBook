"""Estudios de robustez de la sección 10 del protocolo (exploratorios), con puntos de control.

Todos corren sobre los 5 pliegues externos de la repetición 0 de Telco y usan los hiperparámetros que la
validación anidada eligió en cada pliegue, sin volver a optimizar. Las perturbaciones se aplican solo al
pliegue de ajuste; el pliegue de evaluación queda limpio.

Uso: python experiments/confirmatorio/robustez.py [n_procesos]
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import *  # noqa: E402,F403
from correr_anidado import FIJOS, ORIGINAL, guardar, logit, mapem, ruta  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

DS = "telco"
NUM_CRUDAS = ["tenure", "MonthlyCharges", "TotalCharges"]
ESCENARIOS = (
    [f"semilla_{s}" for s in range(10)]
    + ["ruido_x_05", "ruido_x_10", "ruido_x_20", "atipicos_01", "atipicos_05", "ruido_y_05", "ruido_y_10"]
    + ["n_10", "n_25", "n_50", "conforme"]
)


def base_semilla(esc):
    return seed_for("model", int(esc.split("_")[1])) if esc.startswith("semilla_") else None


def perturbar(esc, Xtr, ytr, rng):
    """Devuelve el pliegue de ajuste perturbado y, si aplica, un conjunto de calibración conforme aparte."""
    X, y = Xtr.copy(), ytr.copy()
    for c in NUM_CRUDAS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(float)
    cal = None
    if esc.startswith("ruido_x_") or esc.startswith("atipicos_"):
        frac = int(esc.split("_")[-1]) / 100
        for c in NUM_CRUDAS:
            sd = X[c].std()
            m = rng.random(len(X)) < frac
            if esc.startswith("ruido_x_"):
                X.loc[m, c] = X.loc[m, c] + rng.normal(0, 0.5 * sd, m.sum())
            else:
                X.loc[m, c] = X[c].mean() + rng.choice([-5, 5], m.sum()) * sd
        X["tenure"] = X["tenure"].clip(lower=0)
    elif esc.startswith("ruido_y_"):
        m = rng.random(len(y)) < int(esc.split("_")[-1]) / 100
        y[m] = 1 - y[m]
    elif esc.startswith("n_"):
        idx, _ = train_test_split(np.arange(len(y)), train_size=int(esc.split("_")[1]) / 100, stratify=y, random_state=int(rng.integers(2**31)))
        X, y = X.iloc[np.sort(idx)], y[np.sort(idx)]
    elif esc == "conforme":
        idx, ical = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=int(rng.integers(2**31)))
        cal = (X.iloc[np.sort(ical)], y[np.sort(ical)])
        X, y = X.iloc[np.sort(idx)], y[np.sort(idx)]
    return X, y, cal


def ruta_esc(esc, f, nombre):
    return RESULTADOS / "robustez" / esc / f"f{f}" / f"{nombre}.pkl"


def _datos(esc, f):
    X, y = cargar(DS)
    tr, te = pliegues_externos(y, 0)[f]
    rng = np.random.default_rng(seed_for("test", ESCENARIOS.index(esc), f))
    Xa, ya, cal = perturbar(esc, X.iloc[tr], y[tr], rng)
    return Xa, ya, cal, X.iloc[te], y[te]


def tarea_modelo(esc, f, nombre):
    destino = ruta_esc(esc, f, nombre)
    if destino.exists():
        return
    meta = pickle.load(open(ruta(DS, 0, f, nombre), "rb"))
    Xa, ya, cal, Xte, _ = _datos(esc, f)
    semilla = base_semilla(esc)
    semilla = seed_for("model", 0, f, BENCHMARKS.index(nombre)) if semilla is None else semilla
    oof = np.zeros(len(ya))
    t0 = time.perf_counter()
    for a, b in pliegues_internos(ya, 0, f):
        oof[b] = pipeline_calibrado(DS, nombre, meta["params"], semilla).fit(Xa.iloc[a], ya[a]).predict_proba(Xa.iloc[b])[:, 1]
    final = pipeline_calibrado(DS, nombre, meta["params"], semilla).fit(Xa, ya)
    seg = time.perf_counter() - t0
    guardar(destino, {"oof": oof, "pte": final.predict_proba(Xte)[:, 1], "seg": seg,
                      "pcal": None if cal is None else final.predict_proba(cal[0])[:, 1]})


def tarea_combinar(esc, f):
    destino = ruta_esc(esc, f, "_resumen")
    if destino.exists():
        return
    Xa, ya, cal, Xte, yte = _datos(esc, f)
    exp = {m: pickle.load(open(ruta_esc(esc, f, m), "rb")) for m in BENCHMARKS}
    O = np.column_stack([exp[m]["oof"] for m in BENCHMARKS])
    T = np.column_stack([exp[m]["pte"] for m in BENCHMARKS])
    comb = pickle.load(open(ruta(DS, 0, f, "_combinadores"), "rb"))
    bp = comb[ORIGINAL]["params"]
    pp = preprocesador(DS).fit(Xa)
    Za, Zte = np.asarray(pp.transform(Xa), float), np.asarray(pp.transform(Xte), float)
    sem = base_semilla(esc)
    sem = seed_for("model", 0, f, 100) if sem is None else sem
    m = mapem(Za, ya, O, sem, bp["K"], bp["c"], bp["covarianza"])
    g = mapem(Za, ya, O, sem, 1, 0.0, bp["covarianza"])
    lr = LogisticRegression(C=comb["stacking_logistico"]["params"]["C"], solver="lbfgs", max_iter=3000).fit(logit(O), ya)
    pred = {n: exp[n]["pte"] for n in BENCHMARKS}
    oofs = {n: exp[n]["oof"] for n in BENCHMARKS}
    pred[ORIGINAL] = m.predict_proba_from_predictions(Zte, T)[:, 1]
    pred["mapem_K1"] = g.predict_proba_from_predictions(Zte, T)[:, 1]
    pred["stacking_logistico"] = lr.predict_proba(logit(T))[:, 1]
    pred["promedio_simple"] = T.mean(1)
    # umbral F1: fuera de muestra de los expertos, pesos ajustados en el mismo pliegue (solo se reporta log-loss como principal)
    oofs[ORIGINAL] = m.predict_proba_from_predictions(Za, O)[:, 1]
    oofs["mapem_K1"] = g.predict_proba_from_predictions(Za, O)[:, 1]
    oofs["stacking_logistico"] = lr.predict_proba(logit(O))[:, 1]
    oofs["promedio_simple"] = O.mean(1)
    filas = [{"escenario": esc, "pliegue": f, "modelo": n, **metricas(yte, p, umbral_f1(ya, oofs[n]))} for n, p in pred.items()]
    salida = {"filas": filas, "seg": {n: exp[n]["seg"] for n in BENCHMARKS}}
    if esc == "conforme":
        Zc = np.asarray(pp.transform(cal[0]), float)
        Pc = np.column_stack([exp[n]["pcal"] for n in BENCHMARKS])
        pcal = {n: exp[n]["pcal"] for n in BENCHMARKS}
        pcal[ORIGINAL] = m.predict_proba_from_predictions(Zc, Pc)[:, 1]
        pcal["mapem_K1"] = g.predict_proba_from_predictions(Zc, Pc)[:, 1]
        pcal["stacking_logistico"] = lr.predict_proba(logit(Pc))[:, 1]
        pcal["promedio_simple"] = Pc.mean(1)
        salida["conforme"] = [fila_conforme(n, f, cal[1], pcal[n], yte, pred[n], nivel)
                              for n in pcal for nivel in (0.90, 0.95)]
    guardar(destino, salida)


def fila_conforme(modelo, f, ycal, pcal, yte, pte, nivel):
    """Predicción conforme partida condicionada por clase (Mondrian). No conformidad: 1 - p(clase)."""
    umbrales = {}
    for c in (0, 1):
        s = np.sort(1 - (pcal[ycal == c] if c == 1 else 1 - pcal[ycal == c]))
        k = int(np.ceil((len(s) + 1) * nivel)) - 1
        umbrales[c] = s[min(k, len(s) - 1)]
    en1 = (1 - pte) <= umbrales[1]
    en0 = pte <= umbrales[0]
    cubre = np.where(yte == 1, en1, en0)
    return {"modelo": modelo, "pliegue": f, "nominal": nivel, "cobertura": float(cubre.mean()),
            "cobertura_churn": float(cubre[yte == 1].mean()), "cobertura_no_churn": float(cubre[yte == 0].mean()),
            "tamano_medio": float((en1.astype(int) + en0.astype(int)).mean()),
            "singleton": float(((en1.astype(int) + en0.astype(int)) == 1).mean())}


def sensibilidad(f_r):
    """Curva del log-loss al variar K y c, con las predicciones guardadas de los expertos."""
    r, f = f_r
    X, y = cargar(DS)
    tr, te = pliegues_externos(y, r)[f]
    exp = [pickle.load(open(ruta(DS, r, f, m), "rb")) for m in BENCHMARKS]
    O = np.column_stack([e["oof"] for e in exp])
    T = np.column_stack([e["pte"] for e in exp])
    pp = preprocesador(DS).fit(X.iloc[tr])
    Za, Zt = np.asarray(pp.transform(X.iloc[tr]), float), np.asarray(pp.transform(X.iloc[te]), float)
    cov = pickle.load(open(ruta(DS, r, f, "_combinadores"), "rb"))[ORIGINAL]["params"]["covarianza"]
    filas = []
    for K in range(1, 9):
        for c in (0.0, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0):
            m = mapem(Za, y[tr], O, seed_for("model", r, f, 100), K, c, cov)
            p = np.clip(m.predict_proba_from_predictions(Zt, T)[:, 1], EPS, 1 - EPS)
            filas.append({"rep": r, "pliegue": f, "K": K, "c": c, "log_loss": log_loss(y[te], p)})
    return filas


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    Parallel(n_jobs=n, batch_size=1)(delayed(tarea_modelo)(e, f, m) for e in ESCENARIOS for f in range(5) for m in BENCHMARKS)
    Parallel(n_jobs=n, batch_size=1)(delayed(tarea_combinar)(e, f) for e in ESCENARIOS for f in range(5))
    filas = sum(Parallel(n_jobs=n, batch_size=1)(delayed(sensibilidad)((r, f)) for r in range(5) for f in range(5)), [])
    (RESULTADOS / "robustez").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(RESULTADOS / "robustez" / "sensibilidad_K_c.csv", index=False)
    print("robustez terminada", flush=True)
