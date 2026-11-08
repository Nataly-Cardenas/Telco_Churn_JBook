"""Piloto exploratorio 2: híbrido GLM con splines + boosting sobre el residuo (offset).

Pregunta: ¿un híbrido que parte de la logística con splines (el mejor experto del piloto 1)
y corrige con árboles poco profundos solo lo que la parte aditiva no captura (interacciones)
gana lo suficiente como para un efecto no despreciable (unos 0,0024 de log-loss)?

Mismos datos, pliegues (5x5) y semillas que piloto_oraculo.py. No usa el hold-out.
Se reportan TODAS las configuraciones probadas; la mejor es una cota optimista, porque
se elige mirando el pliegue de evaluación.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from piloto_oraculo import NUMERIC, OUT, load_dev, nadeau_bengio, preprocessor, seed_for

CONFIGS = [  # (profundidad, árboles, tasa)
    (1, 100, 0.05), (2, 100, 0.05), (2, 300, 0.02), (3, 100, 0.03), (3, 300, 0.01),
]
EPS = 1e-6


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run_fold(rep, fold, a, b, X, y, cat):
    seed = seed_for("model", rep, fold)
    Xf, yf, Xe, ye = X.iloc[a], y[a], X.iloc[b], y[b]
    glm = Pipeline([("prep", preprocessor(cat, splines=True)),
                    ("clf", LogisticRegression(C=0.5, max_iter=5000))])
    base = CalibratedClassifierCV(glm, method="sigmoid", cv=3).fit(Xf, yf)
    out = {"glm_splines_platt": ll(ye, base.predict_proba(Xe)[:, 1])}
    glm.fit(Xf, yf)                                     # margen aditivo (offset) del híbrido
    mf, me = glm.decision_function(Xf), glm.decision_function(Xe)
    # control: la misma logística sin Platt, ajustada con el pliegue completo
    out["glm_splines_raw"] = ll(ye, glm.predict_proba(Xe)[:, 1])
    for C in (0.05, 0.2, 2.0):  # control: ¿la ganancia es solo regularización distinta?
        g2 = Pipeline([("prep", preprocessor(cat, splines=True)),
                       ("clf", LogisticRegression(C=C, max_iter=5000))]).fit(Xf, yf)
        out[f"glm_splines_raw_C{C}"] = ll(ye, g2.predict_proba(Xe)[:, 1])
    # control: ¿la ganancia del híbrido es solo una forma no lineal más fina (más nudos)?
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
    for knots in (10, 20):
        pre = ColumnTransformer([
            ("num", Pipeline([("sc", StandardScaler()),
                              ("sp", SplineTransformer(n_knots=knots, degree=3,
                                                       knots="quantile"))]), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat)])
        g3 = Pipeline([("prep", pre), ("clf", LogisticRegression(C=0.5, max_iter=5000))]).fit(Xf, yf)
        out[f"glm_splines_q{knots}"] = ll(ye, g3.predict_proba(Xe)[:, 1])
    prep = preprocessor(cat).fit(Xf)
    Zf, Ze = prep.transform(Xf), prep.transform(Xe)
    for d, n, lr in CONFIGS:
        xgb = XGBClassifier(n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=10,
                            tree_method="hist", n_jobs=1, random_state=seed, eval_metric="logloss")
        xgb.fit(Zf, yf, base_margin=mf)
        p = xgb.predict_proba(Ze, base_margin=me)[:, 1]
        out[f"hybrid_d{d}_n{n}_lr{lr}"] = ll(ye, p)
    return [{"rep": rep, "fold": fold, "method": k, "log_loss": v} for k, v in out.items()]


def main():
    X, y = load_dev()
    cat = [c for c in X.columns if c not in NUMERIC]
    tasks = []
    for rep in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed_for("outer", rep))
        tasks += [(rep, f, a, b) for f, (a, b) in enumerate(skf.split(X, y))]
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=12)(delayed(run_fold)(r, f, a, b, X, y, cat) for r, f, a, b in tasks)
    df = pd.DataFrame([x for r in rows for x in r])
    df.to_csv(OUT / "hibrido_por_pliegue.csv", index=False)
    piv = df.pivot_table(index=["rep", "fold"], columns="method", values="log_loss")
    res = []
    for m in piv.columns:
        d = (piv["glm_splines_platt"] - piv[m]).to_numpy()
        res.append({"method": m, "log_loss": piv[m].mean(), "gain": d.mean(),
                    "folds_better": int((d > 0).sum()),
                    "nb_p": nadeau_bengio(d, 0.2)[1] if m != "glm_splines_platt" else np.nan,
                    "wilcoxon_p": stats.wilcoxon(d).pvalue if m != "glm_splines_platt" else np.nan})
    res = pd.DataFrame(res).sort_values("log_loss")
    res.to_csv(OUT / "hibrido_resumen.csv", index=False)
    print(res.round(4).to_string(index=False))
    print(f"segundos: {time.perf_counter() - t0:.0f}")


if __name__ == "__main__":
    main()
