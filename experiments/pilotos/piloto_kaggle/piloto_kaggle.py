"""Piloto EXPLORATORIO con el dataset del proyecto (WA_Fn-UseC_-Telco-Customer-Churn.csv, versión Kaggle).

- Solo el conjunto de desarrollo (train_index.csv, 5634 clientes). El hold-out (test_index.csv) NO se usa.
- ETL por filas del proyecto (HistoricalFeatures) + StandardScaler + OneHot(drop='first'), ajustados en cada pliegue.
- 5 repeticiones x 5 pliegues externos, semillas seed_for('outer', r) y seed_for('inner', r, f) (raíz 42),
  como en la sección 4.1 del protocolo. La repetición 0 reproduce los pliegues externos del entregable anterior.
- 9 expertos sin SMOTE, calibrados con Platt (cv=3); hiperparámetros fijos (no hay optimización en el piloto).
- Pesos del modelo propio ajustados con predicciones fuera de pliegue (cross-fitting de 3 pliegues): sin fuga.
Uso (desde la raíz del repo): python experiments/pilotos/piloto_kaggle/piloto_kaggle.py . experiments/pilotos/piloto_kaggle/cache
"""
import sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

SEED_ROLES = {'subset': 11, 'outer': 23, 'inner': 37, 'sampler': 41, 'model': 53, 'calibration': 67, 'test': 79, 'optimizer': 83}
def seed_for(role, *c):                       # idéntica a CODIGO_EXPERIMENTO.ipynb
    return int(np.random.SeedSequence([42, SEED_ROLES[role], *c]).generate_state(1)[0])

raw = pd.read_csv(ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"); raw.columns = raw.columns.str.strip()
train_ids = pd.read_csv(ROOT / "train_index.csv")["train_index"].astype(int).to_numpy()
test_ids = pd.read_csv(ROOT / "test_index.csv")["test_index"].astype(int).to_numpy()
assert set(train_ids).isdisjoint(test_ids)
X = raw.loc[train_ids].drop(columns=["customerID", "Churn"]).copy()
y = raw.loc[train_ids, "Churn"].map({"No": 0, "Yes": 1}).astype(int).to_numpy()
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "AvgChargePerMonth"]
SERVICE = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL = [c for c in X.columns if c not in NUMERIC]

class HistoricalFeatures(TransformerMixin, BaseEstimator):   # reglas por fila del proyecto (no aprende nada)
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = X.copy(); total = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if (total.isna() & out["tenure"].ne(0)).any(): raise ValueError("TotalCharges ausente con tenure != 0")
        out["TotalCharges"] = total.fillna(0.0)
        for c in SERVICE: out[c] = out[c].replace("No internet service", "No")
        out["MultipleLines"] = out["MultipleLines"].replace("No phone service", "No")
        avg = out["MonthlyCharges"].to_numpy(float).copy()
        np.divide(out["TotalCharges"].to_numpy(float), out["tenure"].to_numpy(float), out=avg, where=out["tenure"].to_numpy() > 0)
        out["AvgChargePerMonth"] = avg
        return out

def prepro():
    return Pipeline([("features", HistoricalFeatures()), ("prep", ColumnTransformer(
        [("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CATEGORICAL)],
        remainder="drop", sparse_threshold=0.0))])
CONT = [0, 1, 2, 4]                                            # columnas continuas tras el preprocesamiento
def cal(m): return CalibratedClassifierCV(m, method="sigmoid", cv=3)
EXPERTOS = {
    "LR": lambda: cal(LogisticRegression(max_iter=3000)),
    "LR_splines": lambda: cal(Pipeline([("spl", ColumnTransformer([("s", SplineTransformer(n_knots=5), CONT)], remainder="passthrough")),
                                        ("lr", LogisticRegression(C=0.5, max_iter=4000))])),
    "RF": lambda: cal(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=2, random_state=42)),
    "XGB": lambda: cal(XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, n_jobs=2, random_state=42, verbosity=0)),
    "HistGB": lambda: cal(HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=42)),
    "kNN": lambda: cal(KNeighborsClassifier(n_neighbors=50, weights="distance")),
    "CatBoost": lambda: cal(CatBoostClassifier(iterations=400, learning_rate=0.05, depth=4, verbose=0, random_seed=42, thread_count=2)),
    "LightGBM": lambda: cal(LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1)),
    "NaiveBayes": lambda: cal(GaussianNB()),
}
NOM = list(EXPERTOS); json.dump(NOM, open(OUT / "expertos.json", "w"))
t0 = time.time()
for r in range(5):
    outer = StratifiedKFold(5, shuffle=True, random_state=seed_for("outer", r))
    for f, (tr, te) in enumerate(outer.split(X, y)):
        fid = 5 * r + f; dest = OUT / f"cache_r{r}_f{f}.npz"
        if dest.exists(): continue
        pp = prepro().fit(X.iloc[tr]); Xtr, Xte = pp.transform(X.iloc[tr]), pp.transform(X.iloc[te])
        oof = np.zeros((len(tr), len(NOM))); pte = np.zeros((len(te), len(NOM)))
        inner = StratifiedKFold(3, shuffle=True, random_state=seed_for("inner", r, f))
        for j, n in enumerate(NOM):
            for a, b in inner.split(Xtr, y[tr]): oof[b, j] = EXPERTOS[n]().fit(Xtr[a], y[tr][a]).predict_proba(Xtr[b])[:, 1]
            pte[:, j] = EXPERTOS[n]().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        np.savez_compressed(dest, tr=tr, te=te, oof=oof, pte=pte, Xtr=Xtr, Xte=Xte, ytr=y[tr], yte=y[te])
        print(f"r{r} f{f}  {time.time() - t0:.0f}s", flush=True)
np.save(OUT / "y_dev.npy", y)
print("listo")
