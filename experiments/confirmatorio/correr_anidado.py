"""Validación anidada del protocolo (secciones 4 a 7) con puntos de control por tarea.

Tarea de benchmark (conjunto, repetición, pliegue, modelo):
  1. 30 evaluaciones de TPE sobre 3 pliegues internos, minimizando el log-loss del pipeline calibrado.
  2. Las predicciones fuera de pliegue de la mejor evaluación son las entradas de los combinadores.
  3. Se reajusta con todo el pliegue de ajuste externo y se predice el pliegue externo.

Tarea de combinación (conjunto, repetición, pliegue), cuando ya están los 11 benchmarks:
  RegionalMAPEM con 30 evaluaciones de (K, c, covarianza), su ablación y los combinadores estándar.

El proceso se puede interrumpir: al reiniciar salta las tareas ya guardadas.
Uso: python experiments/confirmatorio/correr_anidado.py <conjunto|todos> [n_procesos]
"""
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import *  # noqa: E402,F403
import optuna  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from src.original_model import RegionalMAPEM  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.ERROR)
OPT = ESPACIOS["optimizador"]
N_TRIALS, N_STARTUP = OPT["evaluaciones_por_modelo_y_pliegue_externo"], OPT["arranque_aleatorio"]
N_TRIALS = int(os.environ.get("PRUEBA_N_TRIALS", N_TRIALS))   # solo para la prueba de humo; sin la variable son 30
ORIGINAL = "RegionalMAPEM"
FIJOS = ESPACIOS["modelo_original"]["RegionalMAPEM"]["fijos"]
ESP_ORIGINAL = ESPACIOS["modelo_original"]["RegionalMAPEM"]["espacio"]
ESP_STACK = ESPACIOS["combinadores_estandar"]["stacking_logistico"]["espacio"]


def ruta(ds, r, f, nombre):
    return RESULTADOS / "crudos" / ds / f"r{r}_f{f}" / f"{nombre}.pkl"


def guardar(destino, obj):
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=4)
    tmp.replace(destino)


def estudio(semilla):
    optuna.logging.set_verbosity(optuna.logging.ERROR)     # también en los procesos de trabajo
    return optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=semilla % 2**31, n_startup_trials=N_STARTUP))


def inicio_benchmark(nombre, Xtr, ds):
    ini = dict(ESPACIOS["benchmarks"][nombre]["inicio"])
    if nombre == "svm":      # gamma='scale' se traduce a su valor numérico para que quepa en el rango log
        Z = preprocesador(ds).fit_transform(Xtr)
        lo, hi = ESPACIOS["benchmarks"]["svm"]["espacio"]["gamma"]["low"], ESPACIOS["benchmarks"]["svm"]["espacio"]["gamma"]["high"]
        ini["gamma"] = float(np.clip(1.0 / (Z.shape[1] * Z.var()), lo, hi))
    return ini


def tarea_benchmark(ds, r, f, nombre, X, y, tr, te, idx_modelo):
    """Optimiza un benchmark en un pliegue externo. Devuelve fuera de pliegue, predicción y costos."""
    Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]
    internos = pliegues_internos(ytr, r, f)
    semilla = seed_for("model", r, f, idx_modelo)
    mejor = {"valor": np.inf, "oof": None}
    registro = []

    def objetivo(trial):
        p = sugerir(trial, ESPACIOS["benchmarks"][nombre]["espacio"])
        t0 = time.perf_counter()
        oof = np.zeros(len(ytr))
        for a, b in internos:
            oof[b] = pipeline_calibrado(ds, nombre, p, semilla).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
        pc = np.clip(oof, EPS, 1 - EPS)
        valor = float(np.mean([log_loss(ytr[b], pc[b]) for _, b in internos]))
        registro.append({"params": p, "valor": valor, "seg": time.perf_counter() - t0})
        if valor < mejor["valor"]:
            mejor.update(valor=valor, oof=oof, params=p)
        return valor

    s = estudio(seed_for("optimizer", r, f, idx_modelo))
    s.enqueue_trial(inicio_benchmark(nombre, Xtr, ds))
    t0 = time.perf_counter()
    s.optimize(objetivo, n_trials=N_TRIALS, catch=(Exception,))
    seg_busqueda = time.perf_counter() - t0
    if mejor["oof"] is None:
        raise RuntimeError(f"todas las evaluaciones fallaron: {ds} r{r} f{f} {nombre}")
    fallidos = sum(t.state == optuna.trial.TrialState.FAIL for t in s.trials)
    t0 = time.perf_counter()
    modelo = pipeline_calibrado(ds, nombre, mejor["params"], semilla).fit(Xtr, ytr)
    seg_ajuste = time.perf_counter() - t0
    t0 = time.perf_counter()
    pte = modelo.predict_proba(Xte)[:, 1]
    seg_pred = time.perf_counter() - t0
    return {"modelo": nombre, "params": mejor["params"], "valor_interno": mejor["valor"], "oof": mejor["oof"], "pte": pte,
            "umbral": umbral_f1(ytr, mejor["oof"]), "ensayos": len(s.trials), "fallidos": fallidos, "registro": registro,
            "seg_busqueda": seg_busqueda, "seg_ajuste": seg_ajuste, "seg_pred": seg_pred, "n_pred": len(te)}


def correr_benchmark(ds, r, f, nombre):
    destino = ruta(ds, r, f, nombre)
    if destino.exists():
        return
    X, y = cargar(ds)
    tr, te = pliegues_externos(y, r)[f]
    guardar(destino, tarea_benchmark(ds, r, f, nombre, X, y, tr, te, BENCHMARKS.index(nombre)))
    print(f"{time.strftime('%H:%M:%S')} {ds} r{r} f{f} {nombre}", flush=True)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cruzar(ajustar, internos, n):
    """Predicción fuera de pliegue de un combinador sobre los 3 pliegues internos."""
    oof = np.zeros(n)
    for a, b in internos:
        oof[b] = ajustar(a)(b)
    return oof


def mapem(Z, y, oof, semilla, K, c, cov, duras=False):
    return RegionalMAPEM(n_regions=K, prior_strength=c, covariance=cov, hard_regions=duras, gmm_n_init=FIJOS["n_init_gmm"],
                         max_iter=FIJOS["max_iter_em"], tol=FIJOS["tol_relativa_em"],
                         random_state=semilla).fit_from_predictions(Z, y, oof)


def tarea_combinadores(ds, r, f, Ztr, ytr, Zte, O, T):
    """RegionalMAPEM, ablación y combinadores estándar a partir de las predicciones de los expertos."""
    internos = pliegues_internos(ytr, r, f)
    semilla = seed_for("model", r, f, 100)
    salida = {}

    def cv_mapem(K, c, cov, duras=False):
        def ajustar(a):
            m = mapem(Ztr[a], ytr[a], O[a], semilla, K, c, cov, duras)
            return lambda b: m.predict_proba_from_predictions(Ztr[b], O[b])[:, 1]
        return cruzar(ajustar, internos, len(ytr))

    def valor(oof):
        pc = np.clip(oof, EPS, 1 - EPS)
        return float(np.mean([log_loss(ytr[b], pc[b]) for _, b in internos]))

    registro = []

    def objetivo(trial):
        K = trial.suggest_int("K", ESP_ORIGINAL["K"]["low"], ESP_ORIGINAL["K"]["high"])
        c = trial.suggest_float("c", ESP_ORIGINAL["c"]["low"], ESP_ORIGINAL["c"]["high"], log=True)
        cov = trial.suggest_categorical("covarianza", ESP_ORIGINAL["covarianza"]["valores"])
        t0 = time.perf_counter()
        v = valor(cv_mapem(K, c, cov))
        registro.append({"params": dict(K=K, c=c, covarianza=cov), "valor": v, "seg": time.perf_counter() - t0})
        return v

    s = estudio(seed_for("optimizer", r, f, 100))
    s.enqueue_trial(ESPACIOS["modelo_original"]["RegionalMAPEM"]["inicio"])
    t0 = time.perf_counter()
    s.optimize(objetivo, n_trials=N_TRIALS, catch=(Exception,))
    seg_busqueda = time.perf_counter() - t0
    bp = s.best_params
    K, c, cov = bp["K"], bp["c"], bp["covarianza"]

    def registrar(nombre, oof, pte, extra=None):
        salida[nombre] = {"modelo": nombre, "oof": oof, "pte": pte, "umbral": umbral_f1(ytr, oof), **(extra or {})}

    # modelo completo
    t0 = time.perf_counter()
    m = mapem(Ztr, ytr, O, semilla, K, c, cov)
    seg_ajuste = time.perf_counter() - t0
    t0 = time.perf_counter()
    pte = m.predict_proba_from_predictions(Zte, T)[:, 1]
    seg_pred = time.perf_counter() - t0
    registrar(ORIGINAL, cv_mapem(K, c, cov), pte, dict(
        params=bp, ensayos=len(s.trials), fallidos=sum(t.state == optuna.trial.TrialState.FAIL for t in s.trials),
        registro=registro, seg_busqueda=seg_busqueda, seg_ajuste=seg_ajuste, seg_pred=seg_pred, n_pred=len(pte),
        pesos=m.weights_, pesos_globales=m.global_weights_, convergio=bool(m.converged_), n_iter=int(m.n_iter_),
        min_dif_F=float(np.diff(m.history_).min()), historial=m.history_[:: max(1, len(m.history_) // 200)]))

    # ablación con los hiperparámetros del modelo completo (sección 6)
    sp = mapem(Ztr, ytr, O, semilla, K, 0.0, cov)
    registrar("mapem_sin_prior", cv_mapem(K, 0.0, cov), sp.predict_proba_from_predictions(Zte, T)[:, 1])
    g = mapem(Ztr, ytr, O, semilla, 1, 0.0, cov)
    registrar("mapem_K1", cv_mapem(1, 0.0, cov), g.predict_proba_from_predictions(Zte, T)[:, 1])
    if K > 1:
        d = mapem(Ztr, ytr, O, semilla, K, c, cov, duras=True)
        registrar("mapem_regiones_duras", cv_mapem(K, c, cov, duras=True), d.predict_proba_from_predictions(Zte, T)[:, 1])
    else:   # con una sola región, duras y suaves coinciden
        registrar("mapem_regiones_duras", salida[ORIGINAL]["oof"], pte)
    registrar("promedio_simple", O.mean(1), T.mean(1))

    # stacking logístico: mismo presupuesto (30 evaluaciones de C)
    LO, LT = logit(O), logit(T)

    def cv_stack(C):
        def ajustar(a):
            lr = LogisticRegression(C=C, solver="lbfgs", max_iter=3000).fit(LO[a], ytr[a])
            return lambda b: lr.predict_proba(LO[b])[:, 1]
        return cruzar(ajustar, internos, len(ytr))

    s2 = estudio(seed_for("optimizer", r, f, 101))
    s2.enqueue_trial(ESPACIOS["combinadores_estandar"]["stacking_logistico"]["inicio"])
    s2.optimize(lambda t: valor(cv_stack(sugerir(t, ESP_STACK)["C"])), n_trials=N_TRIALS, catch=(Exception,))
    C = s2.best_params["C"]
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=3000).fit(LO, ytr)
    registrar("stacking_logistico", cv_stack(C), lr.predict_proba(LT)[:, 1], dict(params={"C": C}, ensayos=len(s2.trials)))
    return salida


def correr_combinadores(ds, r, f):
    destino = ruta(ds, r, f, "_combinadores")
    if destino.exists():
        return
    X, y = cargar(ds)
    tr, te = pliegues_externos(y, r)[f]
    exp = [pickle.load(open(ruta(ds, r, f, m), "rb")) for m in BENCHMARKS]
    O = np.column_stack([e["oof"] for e in exp])
    T = np.column_stack([e["pte"] for e in exp])
    pp = preprocesador(ds).fit(X.iloc[tr])
    Ztr, Zte = np.asarray(pp.transform(X.iloc[tr]), float), np.asarray(pp.transform(X.iloc[te]), float)
    guardar(destino, tarea_combinadores(ds, r, f, Ztr, y[tr], Zte, O, T))
    print(f"{time.strftime('%H:%M:%S')} {ds} r{r} f{f} combinadores", flush=True)


def correr(ds, n):
    reps, folds = range(REPETICIONES[ds]), range(5)
    Parallel(n_jobs=n, backend="loky", batch_size=1)(
        delayed(correr_benchmark)(ds, r, f, m) for r in reps for f in folds for m in BENCHMARKS)
    Parallel(n_jobs=n, backend="loky", batch_size=1)(delayed(correr_combinadores)(ds, r, f) for r in reps for f in folds)


if __name__ == "__main__":
    objetivo_ds = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    for ds in (DATASETS if objetivo_ds == "todos" else [objetivo_ds]):
        t0 = time.time()
        correr(ds, n)
        print(f"== {ds} terminado en {(time.time() - t0) / 3600:.2f} h", flush=True)
