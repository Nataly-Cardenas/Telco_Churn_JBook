"""Piezas compartidas del experimento confirmatorio (experiments/protocolo.md).

Fija semillas, carga los 4 conjuntos, construye el preprocesamiento y los 11 benchmarks a partir de
experiments/espacios.json y define las métricas. Nada de este módulo mira resultados.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")       # el paralelismo se hace por tarea, no dentro de cada modelo

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
DATOS = RAIZ / "experiments" / "datos"
RESULTADOS = Path(os.environ.get("RESULTADOS_DIR", RAIZ / "experiments" / "resultados"))
ESPACIOS = json.loads((RAIZ / "experiments" / "espacios.json").read_text(encoding="utf-8"))
BENCHMARKS = list(ESPACIOS["benchmarks"])
EPS = 1e-6
DATASETS = ["telco", "uci_563_iranian", "openml_40701_churn", "openml_1112_kddcup09_churn"]
REPETICIONES = {"telco": 5, "uci_563_iranian": 5, "openml_40701_churn": 5, "openml_1112_kddcup09_churn": 2}  # sección 4 del protocolo
SEED_ROLES = {"subset": 11, "outer": 23, "inner": 37, "sampler": 41, "model": 53, "calibration": 67,
              "test": 79, "optimizer": 83}


def seed_for(role, *coords):
    """Misma derivación de semillas que el entregable anterior (raíz 42)."""
    return int(np.random.SeedSequence([42, SEED_ROLES[role], *coords]).generate_state(1)[0])


# ------------------------------------------------------------------------------- preprocesamiento
class HistoricalFeatures(TransformerMixin, BaseEstimator):
    """ETL por filas de Telco (no aprende nada): TotalCharges, servicios y AvgChargePerMonth."""
    SERVICE = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        total = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if (total.isna() & out["tenure"].ne(0)).any():
            raise ValueError("TotalCharges ausente con tenure != 0")
        out["TotalCharges"] = total.fillna(0.0)
        for c in self.SERVICE:
            out[c] = out[c].replace("No internet service", "No")
        out["MultipleLines"] = out["MultipleLines"].replace("No phone service", "No")
        avg = out["MonthlyCharges"].to_numpy(float).copy()
        np.divide(out["TotalCharges"].to_numpy(float), out["tenure"].to_numpy(float), out=avg,
                  where=out["tenure"].to_numpy() > 0)
        out["AvgChargePerMonth"] = avg
        return out


class FiltroColumnas(TransformerMixin, BaseEstimator):
    """Quita columnas con más de `max_faltantes` de faltantes o sin variación. Se aprende en el pliegue."""

    def __init__(self, max_faltantes=0.95):
        self.max_faltantes = max_faltantes

    def fit(self, X, y=None):
        self.columnas_ = [c for c in X.columns if X[c].isna().mean() <= self.max_faltantes and X[c].nunique() > 1]
        return self

    def transform(self, X):
        return X[self.columnas_]


class _SelectorTipo:
    """Selector de columnas por tipo, serializable (ColumnTransformer lo evalúa dentro del pliegue)."""

    def __init__(self, numericas):
        self.numericas = numericas

    def __call__(self, X):
        return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) == self.numericas]


def preprocesador(ds):
    """Preprocesamiento de cada conjunto. Todo se ajusta dentro del pliegue de entrenamiento."""
    if ds.startswith("telco"):
        num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "AvgChargePerMonth"]
        cat = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
        return Pipeline([("features", HistoricalFeatures()), ("prep", ColumnTransformer(
            [("num", StandardScaler(), num),
             ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat)],
            remainder="drop", sparse_threshold=0.0))])
    pasos = []
    if ds == "openml_1112_kddcup09_churn":
        pasos.append(("filtro", FiltroColumnas(0.95)))
    pasos.append(("prep", ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), _SelectorTipo(True)),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=50,
                                                 sparse_output=False))]), _SelectorTipo(False))],
        remainder="drop", sparse_threshold=0.0)))
    return Pipeline(pasos)


class SplinesContinuas(TransformerMixin, BaseEstimator):
    """B-splines cúbicos solo para las columnas continuas (más de 2 valores en el ajuste).

    Con nudos por cuantiles se quitan los cuantiles repetidos de cada columna, porque las variables de
    conteo tienen muchos empates. Las columnas binarias pasan sin cambio.
    """

    def __init__(self, n_knots=5, knots="uniform", degree=3):
        self.n_knots = n_knots
        self.knots = knots
        self.degree = degree

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.transformadores_ = {}
        for j in range(X.shape[1]):
            x = X[:, j]
            if len(np.unique(x)) <= 2:
                continue
            q = np.linspace(0, 100, self.n_knots)
            grid = np.unique(np.percentile(x, q)) if self.knots == "quantile" else np.linspace(x.min(), x.max(), self.n_knots)
            if len(grid) < 2:
                continue
            self.transformadores_[j] = SplineTransformer(knots=grid[:, None], degree=self.degree,
                                                         extrapolation="constant").fit(x[:, None])
        self.pasan_ = [j for j in range(X.shape[1]) if j not in self.transformadores_]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        partes = [t.transform(X[:, [j]]) for j, t in self.transformadores_.items()]
        return np.hstack(partes + [X[:, self.pasan_]])


# ------------------------------------------------------------------------------- benchmarks
def estimador(nombre, p, semilla):
    """Crea el estimador sin calibrar a partir de los hiperparámetros `p` (más los fijos de espacios.json)."""
    p = {**ESPACIOS["benchmarks"][nombre].get("fijos", {}), **p}
    if nombre == "knn":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"])
    if nombre == "naive_bayes":
        return GaussianNB(var_smoothing=p["var_smoothing"])
    if nombre == "logistic":
        return LogisticRegression(solver=p["solver"], max_iter=p["max_iter"], C=p["C"], penalty=p["penalty"],
                                  random_state=semilla)
    if nombre == "decision_tree":
        return DecisionTreeClassifier(random_state=semilla, **{k: p[k] for k in ("max_depth", "min_samples_leaf", "criterion")})
    if nombre == "random_forest":
        return RandomForestClassifier(random_state=semilla, n_jobs=1, **p)
    if nombre == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=semilla, n_jobs=1, verbosity=0, **p)
    if nombre == "svm":
        return SVC(kernel=p["kernel"], C=p["C"], gamma=p["gamma"], cache_size=500, random_state=semilla)
    if nombre == "logistic_splines":
        return Pipeline([("spl", SplinesContinuas(p["n_knots"], p["knots"], p["degree"])),
                         ("lr", LogisticRegression(C=p["C"], solver=p["solver"], penalty=p["penalty"],
                                                   max_iter=p["max_iter"], random_state=semilla))])
    if nombre == "hist_gb":
        return HistGradientBoostingClassifier(random_state=semilla, **p)
    if nombre == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=semilla, n_jobs=1, **p)
    if nombre == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=semilla, allow_writing_files=False, **p)
    raise ValueError(nombre)


def pipeline_calibrado(ds, nombre, p, semilla):
    """Preprocesamiento + modelo, calibrado con Platt (cv=3). Cada ajuste reajusta el preprocesamiento."""
    return CalibratedClassifierCV(Pipeline([("pre", preprocesador(ds)), ("modelo", estimador(nombre, p, semilla))]),
                                  method="sigmoid", cv=3)


def sugerir(trial, espacio):
    """Traduce el espacio de espacios.json a llamadas de Optuna."""
    p = {}
    for k, v in espacio.items():
        t = v["tipo"]
        if t == "log":
            p[k] = trial.suggest_float(k, v["low"], v["high"], log=True)
        elif t == "float":
            p[k] = trial.suggest_float(k, v["low"], v["high"])
        elif t == "int":
            p[k] = trial.suggest_int(k, v["low"], v["high"])
        elif t == "int_log":
            p[k] = trial.suggest_int(k, v["low"], v["high"], log=True)
        elif t == "categorical":
            p[k] = trial.suggest_categorical(k, v["valores"])
        else:
            raise ValueError(t)
    return p


# ------------------------------------------------------------------------------- datos
def cargar(ds):
    """Devuelve (X, y) del conjunto. En Telco, solo el conjunto de desarrollo (el hold-out no se toca)."""
    if ds == "telco":
        raw = pd.read_csv(RAIZ / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        raw.columns = raw.columns.str.strip()
        ids = pd.read_csv(RAIZ / "train_index.csv")["train_index"].astype(int).to_numpy()
        return raw.loc[ids].drop(columns=["customerID", "Churn"]).reset_index(drop=True), \
            raw.loc[ids, "Churn"].map({"No": 0, "Yes": 1}).astype(int).to_numpy()
    if ds == "openml_40701_churn":
        df = pd.read_csv(DATOS / f"{ds}.csv.gz")
        y = df.pop("class").astype(int).to_numpy()
        df = df.drop(columns=["phone_number"])                 # identificador: un valor distinto por fila
        for c in ("state", "area_code"):                        # códigos, no cantidades
            df[c] = "c" + df[c].astype(str)
        return df, y
    if ds == "uci_563_iranian":
        df = pd.read_csv(DATOS / f"{ds}.csv.gz").drop_duplicates().reset_index(drop=True)  # sin identificador: 300 filas repetidas
        y = df.pop("Churn").astype(int).to_numpy()
        return df, y
    if ds == "openml_1112_kddcup09_churn":
        ruta = DATOS / f"{ds}_10000.csv.gz"
        if not ruta.exists():
            df = pd.read_csv(DATOS / f"{ds}_completo.csv.gz", low_memory=False)
            sub, _ = train_test_split(df, train_size=10_000, stratify=df["class"], random_state=seed_for("subset", 1112) % 2**32)
            sub.to_csv(ruta, index=False, compression={"method": "gzip", "mtime": 0})
        df = pd.read_csv(ruta, low_memory=False)
        y = (df.pop("class") == 1).astype(int).to_numpy()
        return df, y
    raise ValueError(ds)


def cargar_holdout():
    """Solo para la evaluación única final: desarrollo y hold-out de Telco."""
    raw = pd.read_csv(RAIZ / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    raw.columns = raw.columns.str.strip()
    ids = pd.read_csv(RAIZ / "test_index.csv")["test_index"].astype(int).to_numpy()
    return raw.loc[ids].drop(columns=["customerID", "Churn"]).reset_index(drop=True), \
        raw.loc[ids, "Churn"].map({"No": 0, "Yes": 1}).astype(int).to_numpy()


def pliegues_externos(y, r):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed_for("outer", r)).split(np.zeros(len(y)), y))


def pliegues_internos(y, r, f):
    return list(StratifiedKFold(3, shuffle=True, random_state=seed_for("inner", r, f)).split(np.zeros(len(y)), y))


# ------------------------------------------------------------------------------- métricas
def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.mean(idx == b) * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if np.any(idx == b)))


def umbral_f1(y, p):
    """Umbral que maximiza el F1 sobre predicciones fuera de pliegue. Mismo procedimiento para todos."""
    prec, rec, t = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def metricas(y, p, umbral):
    p = np.clip(p, EPS, 1 - EPS)
    return {"log_loss": log_loss(y, p), "f1": f1_score(y, p >= umbral, zero_division=0), "auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p), "ece": ece(y, p), "umbral": umbral}
