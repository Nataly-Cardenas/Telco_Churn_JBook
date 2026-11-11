"""Piloto exploratorio: cota del oráculo para la ponderación por regiones (MAP-EM).

Pregunta: ¿cuánto log-loss podría ganar, como máximo, una combinación convexa de
expertos calibrados con pesos por región, si los pesos se ajustaran sobre el propio
pliegue de evaluación? Si ni ese techo optimista supera con claridad al mejor experto,
la condición R3 no es alcanzable con esta familia de modelos.

Es un análisis EXPLORATORIO previo al protocolo pre-registrado. No usa el hold-out de
prueba (test_index.csv): trabaja solo sobre el conjunto de desarrollo del entregable 3.

Diseño
------
* Datos: conjunto de desarrollo (train_index.csv), mismo preprocesamiento por filas que
  el entregable 3 (HistoricalFeatures) + escalado/one-hot ajustados dentro de cada pliegue.
* Particiones: 5 repeticiones x 5 pliegues estratificados. La repetición 0 usa la semilla
  externa del entregable 3 (seed_for('outer', 0)), por lo que sus pliegues son los mismos.
* Expertos: todos sin SMOTE y calibrados con Platt (CalibratedClassifierCV, cv=3) dentro
  del pliegue de entrenamiento. Hiperparámetros fijos razonables, sin optimización.
* Combinadores honestos: pesos aprendidos con predicciones fuera de pliegue (cross-fitting
  interno de 3 pliegues) del pliegue de entrenamiento.
* Oráculos: pesos ajustados sobre el pliegue de evaluación (cota superior optimista).

Uso:  python -I experiments/pilotos/piloto_oraculo.py [--n-jobs 6] [--repeats 5]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "resultados_oraculo"
EPS = 1e-6
ROOT_SEED = 42
SEED_ROLES = {"outer": 23, "inner": 37, "model": 53}  # mismas claves que el entregable 3
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "AvgChargePerMonth"]
SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]
REGION_COLS = ["tenure", "MonthlyCharges"]  # las 2 entradas de la región (propuesta de Laura)
K_GRID = (2, 4, 6, 8)
C_GRID = (0.0, 10.0, 100.0)


def seed_for(role, *coords):
    """Misma derivación de semillas que el entregable 3."""
    return int(np.random.SeedSequence([ROOT_SEED, SEED_ROLES[role], *coords]).generate_state(1)[0])


# ----------------------------------------------------------------------------- datos
def load_dev():
    raw = pd.read_csv(ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    raw.columns = raw.columns.str.strip()
    train_ids = pd.read_csv(ROOT / "train_index.csv")["train_index"].astype(int).to_numpy()
    X = raw.loc[train_ids].drop(columns=["customerID", "Churn"]).copy()
    y = raw.loc[train_ids, "Churn"].map({"No": 0, "Yes": 1}).astype(int)
    # reglas por fila del entregable 3 (no aprenden parámetros)
    total = pd.to_numeric(X["TotalCharges"], errors="coerce")
    assert not (total.isna() & X["tenure"].ne(0)).any()
    X["TotalCharges"] = total.fillna(0.0)
    for col in SERVICE_COLS:
        X[col] = X[col].replace("No internet service", "No")
    X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")
    t = X["tenure"].to_numpy(float)
    X["AvgChargePerMonth"] = np.where(t > 0, X["TotalCharges"].to_numpy(float) / np.maximum(t, 1),
                                      X["MonthlyCharges"].to_numpy(float))
    return X.reset_index(drop=True), y.reset_index(drop=True).to_numpy()


def preprocessor(categorical, splines=False):
    num = (Pipeline([("sc", StandardScaler()), ("sp", SplineTransformer(n_knots=5, degree=3))])
           if splines else StandardScaler())
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical),
    ], remainder="drop", sparse_threshold=0.0)


def experts(categorical, seed):
    """Benchmarks del entregable 3 sin SMOTE + logística con splines + HistGB."""
    base = {
        "logistic": LogisticRegression(C=1.0, max_iter=3000),
        "logistic_splines": LogisticRegression(C=0.5, max_iter=5000),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=50, weights="distance"),
        "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                                                n_jobs=1, random_state=seed),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=1,
                                 eval_metric="logloss", random_state=seed),
        "hist_gb": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                                  l2_regularization=1.0, early_stopping=False,
                                                  random_state=seed),
        "svm": SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed),
    }
    out = {}
    for name, est in base.items():
        pipe = Pipeline([("prep", preprocessor(categorical, splines=name == "logistic_splines")),
                         ("clf", est)])
        out[name] = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    return out


# ----------------------------------------------------------------------- MAP-EM (numpy)
def map_em(P, y, R, alpha, max_iter=500, tol=1e-9):
    """Pesos por región pi[k, m] de una mezcla lineal de expertos con regiones fijas.

    P: (n, M) probabilidades de churn de los expertos; y: (n,) etiquetas;
    R: (n, K) pertenencias fijas a las regiones; alpha: (K, M) parámetros Dirichlet >= 1.
    Devuelve pi (K, M) y la traza de la log-posterior (debe ser no decreciente).
    """
    P = np.clip(P, EPS, 1 - EPS)
    L = np.where(y[:, None] == 1, P, 1 - P)                     # (n, M) verosimilitud de cada experto
    K, M = R.shape[1], P.shape[1]
    pi = np.full((K, M), 1.0 / M)
    trace = []
    for _ in range(max_iter):
        mix = R[:, :, None] * pi[None] * L[:, None, :]          # (n, K, M)
        tot = mix.sum(axis=(1, 2))
        logpost = np.log(tot).sum() + ((alpha - 1) * np.log(np.maximum(pi, 1e-300))).sum()
        trace.append(logpost)
        if len(trace) > 1 and trace[-1] - trace[-2] < tol * max(1.0, abs(trace[-2])):
            break
        gamma = mix / tot[:, None, None]                        # paso E
        num = gamma.sum(axis=0) + alpha - 1                      # paso M (forma cerrada)
        pi = num / num.sum(axis=1, keepdims=True)
    return pi, np.array(trace)


def combine(P, R, pi):
    return np.clip(np.einsum("nk,km,nm->n", R, pi, P), EPS, 1 - EPS)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------ un pliegue
def run_fold(rep, fold, fit_idx, ev_idx, X, y, categorical):
    t0 = time.perf_counter()
    seed = seed_for("model", rep, fold)
    Xf, yf, Xe, ye = X.iloc[fit_idx], y[fit_idx], X.iloc[ev_idx], y[ev_idx]
    inner = StratifiedKFold(3, shuffle=True, random_state=seed_for("inner", rep, fold))
    names, P_oof, P_ev = [], [], []
    for name, model in experts(categorical, seed).items():
        P_oof.append(cross_val_predict(clone(model), Xf, yf, cv=inner, method="predict_proba")[:, 1])
        P_ev.append(clone(model).fit(Xf, yf).predict_proba(Xe)[:, 1])
        names.append(name)
    P_oof, P_ev = np.column_stack(P_oof), np.column_stack(P_ev)
    M = len(names)

    # regiones: GMM sobre 2 entradas estandarizadas con media/desv del pliegue de ajuste
    Zf = Xf[REGION_COLS].to_numpy(float)
    mu, sd = Zf.mean(0), Zf.std(0)
    Zf, Ze = (Zf - mu) / sd, (Xe[REGION_COLS].to_numpy(float) - mu) / sd

    preds = {f"expert:{n}": P_ev[:, j] for j, n in enumerate(names)}
    preds["avg_simple"] = P_ev.mean(1)
    ones_f, ones_e = np.ones((len(yf), 1)), np.ones((len(ye), 1))
    g, _ = map_em(P_oof, yf, ones_f, np.ones((1, M)))
    preds["em_global"] = combine(P_ev, ones_e, g)
    stack = LogisticRegression(C=1.0, max_iter=2000).fit(logit(P_oof), yf)
    preds["stacking_logit"] = stack.predict_proba(logit(P_ev))[:, 1]

    gmm_info, mono = {}, True
    for K in K_GRID:
        gmm = GaussianMixture(K, covariance_type="full", random_state=seed, n_init=3).fit(Zf)
        Rf, Re = gmm.predict_proba(Zf), gmm.predict_proba(Ze)
        gmm_info[K] = Rf.mean(0).round(3).tolist()
        for c in C_GRID:
            alpha = 1.0 + c * np.tile(g, (K, 1))
            pi, tr = map_em(P_oof, yf, Rf, alpha)
            mono &= bool(np.all(np.diff(tr) >= -1e-8 * np.abs(tr[:-1])))
            preds[f"mapem_K{K}_c{int(c)}"] = combine(P_ev, Re, pi)
        # oráculo: pesos por región ajustados sobre el pliegue de evaluación (sin prior)
        pi_or, _ = map_em(P_ev, ye, Re, np.ones((K, M)))
        preds[f"oracle_region_K{K}"] = combine(P_ev, Re, pi_or)
        # oráculo duro: mejor experto por región (asignación dura), elegido en evaluación
        hard = Re.argmax(1)
        best = np.empty(len(ye))
        for k in np.unique(hard):
            msk = hard == k
            losses = [log_loss(ye[msk], P_ev[msk, j], labels=[0, 1]) for j in range(M)]
            best[msk] = P_ev[msk, int(np.argmin(losses))]
        preds[f"oracle_best_expert_K{K}"] = best
    pi_g_or, _ = map_em(P_ev, ye, ones_e, np.ones((1, M)))
    preds["oracle_global"] = combine(P_ev, ones_e, pi_g_or)
    st_or = LogisticRegression(C=1e4, max_iter=5000).fit(logit(P_ev), ye)
    preds["oracle_stacking_logit"] = st_or.predict_proba(logit(P_ev))[:, 1]

    rows, inst = [], {}
    for k, p in preds.items():
        p = np.clip(p, EPS, 1 - EPS)
        ll_i = -(ye * np.log(p) + (1 - ye) * np.log(1 - p))
        inst[k] = ll_i
        rows.append({"rep": rep, "fold": fold, "method": k, "log_loss": ll_i.mean(),
                     "brier": np.mean((p - ye) ** 2), "auc": roc_auc_score(ye, p)})
    corr = np.corrcoef(logit(P_ev).T)
    return {"rows": rows, "inst": inst, "ev_idx": ev_idx, "mono": mono, "gmm": gmm_info,
            "names": names, "mean_corr": float(corr[np.triu_indices(M, 1)].mean()),
            "secs": time.perf_counter() - t0}


# --------------------------------------------------------------------- estadística
def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sign(a[:, None] - b[None, :]).mean())


def nadeau_bengio(d, test_frac):
    """t corregida de Nadeau-Bengio sobre diferencias por pliegue (validación repetida)."""
    J = len(d)
    var = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / J + test_frac / (1 - test_frac)) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def paired_perm(la, lb, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = la - lb
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(1)
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    X, y = load_dev()
    categorical = [c for c in X.columns if c not in NUMERIC]
    tasks = []
    for rep in range(args.repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed_for("outer", rep))
        for fold, (a, b) in enumerate(skf.split(X, y)):
            tasks.append((rep, fold, a, b))
    t0 = time.perf_counter()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_fold)(r, f, a, b, X, y, categorical) for r, f, a, b in tasks)
    wall = time.perf_counter() - t0

    df = pd.DataFrame([row for r in res for row in r["rows"]])
    df.to_csv(OUT / "metricas_por_pliegue.csv", index=False)
    summ = (df.groupby("method")[["log_loss", "brier", "auc"]].agg(["mean", "std"])
            .sort_values(("log_loss", "mean")))
    summ.to_csv(OUT / "resumen.csv")

    piv = df.pivot_table(index=["rep", "fold"], columns="method", values="log_loss")
    experts_cols = [c for c in piv.columns if c.startswith("expert:")]
    best_expert = piv[experts_cols].mean().idxmin()
    honest = [c for c in piv.columns if not c.startswith(("expert:", "oracle"))]
    best_honest = piv[honest].mean().idxmin()
    test_frac = 0.2

    comps = []
    targets = [best_honest, "em_global", "stacking_logit", "mapem_K4_c10"] + \
              [c for c in piv.columns if c.startswith("oracle")]
    for m in dict.fromkeys(targets):                         # sin duplicados, conserva el orden
        d = (piv[best_expert] - piv[m]).to_numpy()            # >0: m mejora
        w = stats.wilcoxon(piv[best_expert], piv[m])
        t, p_nb = nadeau_bengio(d, test_frac)
        # permutación pareada sobre pérdidas por instancia, repetición 0 (cada cliente 1 vez)
        la = np.concatenate([r["inst"][best_expert] for r in res if r["rows"][0]["rep"] == 0])
        lb = np.concatenate([r["inst"][m] for r in res if r["rows"][0]["rep"] == 0])
        _, p_perm = paired_perm(la, lb)
        comps.append({"method": m, "vs": best_expert, "mean_gain_logloss": d.mean(),
                      "sd_gain": d.std(ddof=1), "folds_better": int((d > 0).sum()),
                      "cliff_delta_folds": -cliffs_delta(piv[m], piv[best_expert]),
                      "wilcoxon_p": w.pvalue, "nb_t": t, "nb_p": p_nb, "perm_p_rep0": p_perm})
    comp = pd.DataFrame(comps)
    comp.to_csv(OUT / "comparaciones_vs_mejor_experto.csv", index=False)

    # ¿qué ganancia media haría falta para |delta de Cliff| >= 0.147 con esta dispersión?
    sd_fold = piv[best_expert].std(ddof=1)
    from scipy.stats import norm
    d_needed = np.sqrt(2) * norm.ppf((1 + 0.147) / 2)        # aprox. normal, varianzas iguales
    meta = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"), "n_dev": int(len(y)), "churn_rate": float(y.mean()),
        "repeats": args.repeats, "folds": 5, "experts": res[0]["names"],
        "best_expert": best_expert, "best_honest_combiner": best_honest,
        "em_monotone_all_folds": bool(all(r["mono"] for r in res)),
        "mean_pairwise_logit_corr": float(np.mean([r["mean_corr"] for r in res])),
        "sd_logloss_best_expert_folds": float(sd_fold),
        "gain_needed_for_cliff_0147_approx": float(d_needed * sd_fold),
        "gmm_region_shares_rep0_fold0": res[0]["gmm"],
        "wall_seconds": wall, "fold_seconds_mean": float(np.mean([r["secs"] for r in res])),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string())
    print(comp.round(4).to_string())
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
