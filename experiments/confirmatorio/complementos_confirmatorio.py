"""Evaluaciones descriptivas y exploratorias que dependen de la validación anidada ya terminada.

  dashboard    : las 7 combinaciones del entregable 3 (secciones 4 y 9 del protocolo) en los mismos 25 pliegues.
  escalabilidad: tiempo de ajuste al variar n y p con datos sintéticos, y memoria y tamaño por modelo en Telco.
  shap         : SHAP del modelo original y del mejor benchmark (repetición 0, pliegue 0).

Uso: python experiments/confirmatorio/complementos_confirmatorio.py dashboard|escalabilidad|shap [n_procesos]
Escalabilidad conviene correrla con 1 proceso: con varios en paralelo los tiempos de ajuste se contaminan.
"""
import json
import pickle
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import *  # noqa: E402,F403
from correr_anidado import ORIGINAL, mapem, ruta  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

SAL = RESULTADOS / "tablas"
SAL.mkdir(parents=True, exist_ok=True)
COMBINACIONES = {"random_forest": "random_forest__smote__bayesian_tpe", "xgboost": "xgboost__smote__random",
                 "logistic": "logistic__smote__grid", "svm": "svm__smote__genetic", "naive_bayes": "naive_bayes__none__random",
                 "knn": "knn__smote__bayesian_tpe", "decision_tree": "decision_tree__smote__bayesian_tpe"}


def modelo_e3(nombre, p, semilla):
    from xgboost import XGBClassifier
    if nombre == "random_forest":
        return RandomForestClassifier(random_state=semilla, n_jobs=1, **p)
    if nombre == "xgboost":
        return XGBClassifier(tree_method="hist", random_state=semilla, n_jobs=1, verbosity=0, **p)
    if nombre == "logistic":
        return LogisticRegression(solver="liblinear", max_iter=3000, random_state=semilla, **p)
    if nombre == "svm":
        return CalibratedClassifierCV(SVC(kernel="rbf", random_state=semilla, **p), method="sigmoid", cv=3)
    if nombre == "naive_bayes":
        return GaussianNB(**p)
    if nombre == "knn":
        return KNeighborsClassifier(**p)
    return DecisionTreeClassifier(random_state=semilla, **p)


def una_combinacion(nombre, r, f):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as PipeI
    X, y = cargar("telco")
    tr, te = pliegues_externos(y, r)[f]
    u = pd.read_csv(RAIZ / "resultados_modelado" / "analisis_final" / "unidades.csv")
    eid = COMBINACIONES[nombre]
    p = json.loads(u[(u.experiment_id == eid) & (u.replicate == 0) & (u.outer_fold == f)].best_params.iloc[0])
    semilla = seed_for("model", r, f, BENCHMARKS.index(nombre))
    # imblearn no admite un Pipeline como paso intermedio: se aplanan los pasos del preprocesador
    pasos = list(preprocesador("telco").steps)
    if "__smote__" in eid:
        pasos.append(("smote", SMOTE(random_state=semilla % 2**31)))
    pasos.append(("modelo", modelo_e3(nombre, p, semilla)))
    prob = PipeI(pasos).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return {"combinacion": eid, "modelo": nombre, "rep": r, "pliegue": f, **metricas(y[te], prob, 0.5)}


def escala_uno(nombre, n, p):
    from sklearn.datasets import make_classification
    Xs, ys = make_classification(n_samples=n, n_features=p, n_informative=max(2, p // 2), random_state=0)
    X = pd.DataFrame(Xs, columns=[f"x{j}" for j in range(p)])
    q = ESPACIOS["benchmarks"][nombre]["inicio"]
    if nombre == "svm":
        q = {**q, "gamma": 1.0 / p}
    t0 = time.perf_counter()
    pipeline_calibrado("sintetico", nombre, q, 0).fit(X, ys)
    return {"modelo": nombre, "n": n, "p": p, "seg_ajuste": time.perf_counter() - t0}


def escala_original(n, p, M=11, K=4):
    from sklearn.datasets import make_classification
    Xs, ys = make_classification(n_samples=n, n_features=p, n_informative=max(2, p // 2), random_state=0)
    rng = np.random.default_rng(0)
    base = np.clip(0.2 + 0.6 * ys, EPS, 1 - EPS)
    O = np.clip(base[:, None] + rng.normal(0, 0.2, (n, M)), 0.01, 0.99)
    t0 = time.perf_counter()
    m = mapem(Xs, ys, O, 0, K, 10.0, "full")
    return {"modelo": ORIGINAL + " (solo regiones y pesos)", "n": n, "p": p, "seg_ajuste": time.perf_counter() - t0, "iter_em": int(m.n_iter_)}


def memoria(nombre, X, y, tr, te, params):
    tracemalloc.start()
    t0 = time.perf_counter()
    m = pipeline_calibrado("telco", nombre, params, 0).fit(X.iloc[tr], y[tr])
    seg = time.perf_counter() - t0
    pico = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    t0 = time.perf_counter()
    m.predict_proba(X.iloc[te])
    pred = time.perf_counter() - t0
    return {"modelo": nombre, "seg_ajuste": seg, "seg_pred_por_1000": 1000 * pred / len(te), "pico_python_mb": pico / 1e6,
            "tamano_serializado_mb": len(pickle.dumps(m)) / 1e6}


def _shap_trozo(f, fondo, trozo, semilla):
    """KernelExplainer muestrea coaliciones con np.random: la semilla fija de cada trozo lo hace reproducible."""
    import shap
    np.random.seed(semilla)
    return shap.KernelExplainer(f, fondo, link="identity").shap_values(trozo, nsamples=300, silent=True)


def correr_shap(n):
    X, y = cargar("telco")
    tr, te = pliegues_externos(y, 0)[0]
    res = json.loads((SAL / "veredicto_R3.json").read_text(encoding="utf-8"))
    mejor = res["mejor_benchmark"]["telco"]
    com = pickle.load(open(ruta("telco", 0, 0, "_combinadores"), "rb"))[ORIGINAL]
    exp = {m: pickle.load(open(ruta("telco", 0, 0, m), "rb")) for m in BENCHMARKS}
    finales = Parallel(n_jobs=min(n, 11))(delayed(lambda m: pipeline_calibrado("telco", m, exp[m]["params"], seed_for("model", 0, 0, BENCHMARKS.index(m))).fit(X.iloc[tr], y[tr]))(m) for m in BENCHMARKS)
    O = np.column_stack([exp[m]["oof"] for m in BENCHMARKS])
    pp = preprocesador("telco").fit(X.iloc[tr])
    Ztr = np.asarray(pp.transform(X.iloc[tr]), float)
    bp = com["params"]
    mo = mapem(Ztr, y[tr], O, seed_for("model", 0, 0, 100), bp["K"], bp["c"], bp["covarianza"])
    columnas = list(X.columns)
    rng = np.random.default_rng(seed_for("test", 900))
    fondo = X.iloc[rng.choice(tr, 100, replace=False)]
    muestra = X.iloc[rng.choice(te, 200, replace=False)]

    def como_df(A):
        d = pd.DataFrame(A, columns=columnas)
        for c in columnas:
            d[c] = d[c].astype(X[c].dtype)
        return d

    def f_original(A):
        d = como_df(A)
        T = np.column_stack([e.predict_proba(d)[:, 1] for e in finales])
        return mo.predict_proba_from_predictions(np.asarray(pp.transform(d), float), T)[:, 1]

    def f_mejor(A):
        return finales[BENCHMARKS.index(mejor)].predict_proba(como_df(A))[:, 1]

    salida = {}
    F, M = fondo.to_numpy(dtype=object), muestra.to_numpy(dtype=object)
    trozos = np.array_split(np.arange(len(M)), 10)
    for j, (nombre, f) in enumerate(((ORIGINAL, f_original), (mejor, f_mejor))):
        partes = Parallel(n_jobs=n, batch_size=1)(delayed(_shap_trozo)(f, F, M[t], seed_for("test", 901, j, i) % 2**32)
                                                 for i, t in enumerate(trozos))
        sv = np.concatenate([np.asarray(s) for s in partes], axis=0)
        salida[nombre] = np.abs(sv).mean(0)
        np.save(SAL / f"shap_valores_{nombre}.npy", sv)
    pd.DataFrame(salida, index=columnas).to_csv(SAL / "shap_importancia.csv")
    regiones = pd.DataFrame(mo.weights_, columns=BENCHMARKS)
    regiones.index.name = "region"
    regiones.to_csv(SAL / "pesos_por_region_r0_f0.csv")
    print("shap listo", mejor)


if __name__ == "__main__":
    cual = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if cual == "dashboard":
        filas = Parallel(n_jobs=n, batch_size=1)(delayed(una_combinacion)(m, r, f) for m in COMBINACIONES for r in range(5) for f in range(5))
        pd.DataFrame(filas).to_csv(SAL / "combinaciones_dashboard.csv", index=False)
    elif cual == "escalabilidad":
        ns, ps = (1000, 2000, 4000, 8000), (10, 20, 40, 80)
        tareas = [(m, n, 20) for m in BENCHMARKS for n in ns] + [(m, 2000, p) for m in BENCHMARKS for p in ps]
        filas = Parallel(n_jobs=n, batch_size=1)(delayed(escala_uno)(*t) for t in tareas)
        filas += [escala_original(n, 20) for n in ns] + [escala_original(2000, p) for p in ps]
        pd.DataFrame(filas).to_csv(SAL / "escalabilidad.csv", index=False)
        X, y = cargar("telco")
        tr, te = pliegues_externos(y, 0)[0]
        filas = [memoria(m, X, y, tr, te, pickle.load(open(ruta("telco", 0, 0, m), "rb"))["params"]) for m in BENCHMARKS]
        pd.DataFrame(filas).to_csv(SAL / "memoria_por_modelo.csv", index=False)
    elif cual == "shap":
        correr_shap(n)
    print("listo", cual, flush=True)
