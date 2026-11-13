"""Tablas del paper (filas LaTeX) generadas desde las tablas versionadas del experimento.

Uso (desde la raíz del repositorio): python paper/tables/make_tables.py
Escribe un fragmento .tex por tabla en paper/tables/ (se insertan con \\input).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("REPO_ROOT", os.getcwd())
OUT = os.environ.get("TAB_OUT", os.path.join(ROOT, "paper", "tables"))
TAB = os.path.join(ROOT, "experiments", "resultados", "tablas")
os.makedirs(OUT, exist_ok=True)

rm = pd.read_csv(os.path.join(TAB, "resumen_modelos.csv"))
fam = pd.read_csv(os.path.join(TAB, "familia_resultados.csv"))
mpp = pd.read_csv(os.path.join(TAB, "metricas_por_pliegue.csv"))
bay = pd.read_csv(os.path.join(TAB, "bayesiana_rope.csv"))
cf = pd.read_csv(os.path.join(TAB, "libro", "corridas_finales.csv"))
cj = pd.read_csv(os.path.join(TAB, "libro", "conjuntos.csv")).set_index("conjunto")
cost = pd.read_csv(os.path.join(TAB, "costos_por_pliegue.csv"))
mem = pd.read_csv(os.path.join(TAB, "memoria_por_modelo.csv")).set_index("modelo")

N = {"knn": "kNN", "naive_bayes": "Naive Bayes", "logistic": "Regresión logística",
     "decision_tree": "Árbol de decisión", "random_forest": "Random Forest", "xgboost": "XGBoost",
     "svm": "SVM (RBF)", "logistic_splines": "Logística con splines", "hist_gb": "HistGB",
     "lightgbm": "LightGBM", "catboost": "CatBoost", "RegionalMAPEM": r"\textbf{RegionalMAPEM}",
     "mapem_sin_prior": r"Sin prior ($c=0$)", "mapem_K1": r"Sin regiones ($K=1$)",
     "mapem_regiones_duras": "Regiones duras", "promedio_simple": "Promedio simple",
     "stacking_logistico": "Stacking logístico"}
BENCH = ["knn", "naive_bayes", "logistic", "decision_tree", "random_forest", "xgboost", "svm",
         "logistic_splines", "hist_gb", "lightgbm", "catboost"]
ABL = ["mapem_sin_prior", "mapem_K1", "mapem_regiones_duras", "promedio_simple", "stacking_logistico"]
BEST = {"telco": "logistic_splines", "uci_563_iranian": "hist_gb", "openml_40701_churn": "catboost",
        "openml_1112_kddcup09_churn": "hist_gb"}
DSN = {"telco": "Telco (Kaggle)", "uci_563_iranian": "Iranian (UCI 563)",
       "openml_40701_churn": "churn (OpenML 40701)", "openml_1112_kddcup09_churn": "KDDCup09 (OpenML 1112)"}


def f(x, d=4):
    return f"{x:.{d}f}".replace(".", "{,}").replace("-", "$-$")


def pf(p):
    if p <= 0:
        return r"$<10^{-15}$"
    if p >= 0.995:
        return "1"
    if p >= 0.001:
        return f"{p:.3f}".replace(".", "{,}")
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$".replace(".", "{,}")


def bold_if(p):
    return r"\textbf{" + pf(p) + "}" if p < 0.05 else pf(p)


def test(ds, rival, prueba):
    x = fam[(fam.dataset == ds) & (fam.metrica == "log_loss") & (fam.modelo_b == rival) & (fam.prueba == prueba)]
    return x.iloc[0]


def dz(ds, a, b):
    x = mpp[(mpp.conjunto == ds) & (mpp.modelo == a)].set_index(["rep", "pliegue"]).log_loss
    y = mpp[(mpp.conjunto == ds) & (mpp.modelo == b)].set_index(["rep", "pliegue"]).log_loss
    d = (x - y).dropna()
    return d.mean() / d.std(ddof=1)


def write(name, lines):
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# Tabla de desempeño en Telco (media ± de de los 25 pliegues externos)
t = rm[rm.conjunto == "telco"].set_index("modelo")
order = BENCH + ["RegionalMAPEM"] + ABL
lo_better = {"log_loss", "brier", "ece"}
best = {c: (t.loc[order, c + "_media"].idxmin() if c in lo_better else t.loc[order, c + "_media"].idxmax())
        for c in ["log_loss", "brier", "auc", "f1", "ece"]}
L = []
for m in order:
    if m in ("RegionalMAPEM", ABL[0]):
        L.append(r"\midrule")
    cells = [N[m]]
    for c in ["log_loss", "brier", "auc", "f1", "ece"]:
        s = f"{f(t.loc[m, c + '_media'])} $\\pm$ {f(t.loc[m, c + '_de'])}"
        cells.append(r"\textbf{" + s + "}" if best[c] == m else s)
    L.append(" & ".join(cells) + r" \\")
write("tab_desempeno", L)

# Tabla de pruebas frente a benchmarks y ablación (Telco, log-loss)
b = bay[bay.conjunto == "telco"].set_index("rival")
L = []
for rv in BENCH + ABL:
    if rv == ABL[0]:
        L.append(r"\midrule")
    t1, t2, t4, t3 = (test("telco", rv, p) for p in ["T1", "T2", "T4", "T3_nemenyi"])
    bb = b.loc[rv]
    L.append(" & ".join([
        N[rv], f"{f(t1.dif_media * 1e3, 2)} [{f(t1.ic_bajo * 1e3, 2)}; {f(t1.ic_alto * 1e3, 2)}]",
        bold_if(t1.p_holm), bold_if(t2.p_holm), bold_if(t4.p_holm), bold_if(t3.p_holm),
        f(t1.cliff, 3), f(dz("telco", "RegionalMAPEM", rv), 2),
        f"{bb.p_original_mejor:.2f}/{bb.p_equivalentes:.2f}/{bb.p_rival_mejor:.2f}".replace(".", "{,}"),
    ]) + r" \\")
write("tab_pruebas", L)

# Tabla de conjuntos de datos y comparación con el mejor benchmark de cada uno
L = []
for ds, bm in BEST.items():
    c = cj.loc[ds]
    r = rm[rm.conjunto == ds].set_index("modelo")
    t1, t2, t4 = (test(ds, bm, p) for p in ["T1", "T2", "T4"])
    L.append(" & ".join([
        DSN[ds], f"{int(c.n)}", f"{int(c.p_original)}/{int(c.p_preprocesado)}", f(100 * c.churn, 1),
        f"{int(c.repeticiones)}$\\times$5", N[bm], f(r.loc[bm, "log_loss_media"]),
        f(r.loc["RegionalMAPEM", "log_loss_media"]), f(r.loc["stacking_logistico", "log_loss_media"]),
        " / ".join(bold_if(p) for p in [t1.p_holm, t2.p_holm, t4.p_holm]),
        f(t1.cliff, 3), f(dz(ds, "RegionalMAPEM", bm), 2),
    ]) + r" \\")
write("tab_conjuntos", L)

# Tabla del hold-out y del cambio de distribución por antigüedad
L = []
for m in ["RegionalMAPEM", "mapem_sin_prior", "mapem_K1", "stacking_logistico", "logistic_splines",
          "hist_gb", "lightgbm", "xgboost", "catboost", "random_forest", "logistic"]:
    h = cf[(cf.corrida == "holdout") & (cf.modelo == m)].iloc[0]
    s = cf[(cf.corrida == "tenure") & (cf.modelo == m)].iloc[0]
    L.append(" & ".join([N[m], f(h.log_loss), f(h.brier), f(h.auc), f(h.f1, 3), f(h.ece, 3),
                         f(s.log_loss), f(s.ece, 3)]) + r" \\")
write("tab_holdout", L)

# Tabla de costo computacional (Telco, mediana por pliegue externo)
c = cost[cost.conjunto == "telco"].groupby("modelo").agg(
    {"ensayos": "sum", "fallidos": "sum", "seg_busqueda": "median", "seg_ajuste": "median", "seg_pred_por_1000": "median"})
L = []
for m in [x for x in BENCH + ["RegionalMAPEM"] + ABL if x in c.index]:
    mb = f(mem.loc[m, "tamano_serializado_mb"], 2) if m in mem.index else "---"
    L.append(" & ".join([N[m], f"{int(c.loc[m, 'ensayos'])}", f"{int(c.loc[m, 'fallidos'])}",
                         f(c.loc[m, "seg_busqueda"], 1), f(c.loc[m, "seg_ajuste"], 2),
                         f(c.loc[m, "seg_pred_por_1000"], 3), mb]) + r" \\")
write("tab_costo", L)
print("modelos con costo:", list(c.index))
