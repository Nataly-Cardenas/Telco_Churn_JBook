"""Figuras del paper (RegionalMAPEM). Un solo script, PDF vectorial, texto en inglés.

Uso (desde la raíz del repositorio Telco_Churn_JBook):
    python paper/figures/make_figures.py
Lee solo tablas versionadas de experiments/resultados/ y escribe en paper/figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = os.environ.get("REPO_ROOT", os.getcwd())
OUT = os.environ.get("FIG_OUT", os.path.join(ROOT, "paper", "figures"))
TAB = os.path.join(ROOT, "experiments", "resultados", "tablas")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
    "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "lines.linewidth": 2.0, "axes.grid": True, "grid.alpha": 0.3,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})
W = 6.3  # ancho útil de la página (pulgadas)
FOCAL, GREY, DARK = "#1F3A5F", "#9AA5B1", "#333333"

NAMES = {"knn": "kNN", "naive_bayes": "Naive Bayes", "logistic": "Logistic regression",
         "decision_tree": "Decision tree", "random_forest": "Random forest", "xgboost": "XGBoost",
         "svm": "SVM (RBF)", "logistic_splines": "Spline logistic", "hist_gb": "HistGB",
         "lightgbm": "LightGBM", "catboost": "CatBoost", "RegionalMAPEM": "RegionalMAPEM",
         "mapem_sin_prior": "No prior (c = 0)", "mapem_K1": "No regions (K = 1)",
         "mapem_regiones_duras": "Hard regions", "promedio_simple": "Simple average",
         "stacking_logistico": "Logistic stacking"}
BENCH = ["knn", "naive_bayes", "logistic", "decision_tree", "random_forest", "xgboost", "svm",
         "logistic_splines", "hist_gb", "lightgbm", "catboost"]
ABL = ["mapem_sin_prior", "mapem_K1", "mapem_regiones_duras", "promedio_simple", "stacking_logistico"]
DS = {"telco": "Telco", "uci_563_iranian": "Iranian", "openml_40701_churn": "OpenML 40701",
      "openml_1112_kddcup09_churn": "KDDCup09"}

fam = pd.read_csv(os.path.join(TAB, "familia_resultados.csv"))
mpp = pd.read_csv(os.path.join(TAB, "metricas_por_pliegue.csv"))


def n_rejections(ds, rival, blocks):
    f = fam[(fam.dataset == ds) & (fam.metrica == "log_loss") & (fam.modelo_b == rival) & fam.bloque.isin(blocks)]
    return int(((f.p_holm < 0.05) & (f.dif_media < 0)).sum())


def t1_row(ds, rival):
    f = fam[(fam.dataset == ds) & (fam.metrica == "log_loss") & (fam.modelo_b == rival) & (fam.prueba == "T1")]
    return f.iloc[0]


# ------------------------------------------------------------------ Figura 2: diferencias pareadas
def fig_differences():
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 7.6), gridspec_kw={"height_ratios": [16, 4.6], "hspace": 0.42})
    rows = []
    for r in BENCH:
        t = t1_row("telco", r)
        k = n_rejections("telco", r, ["A1_principal_benchmarks", "A3_principal_friedman"])
        rows.append(("bench", r, t.dif_media, t.ic_bajo, t.ic_alto, k))
    for r in ABL:
        t = t1_row("telco", r)
        k = n_rejections("telco", r, ["A2_principal_ablacion", "A3_principal_friedman"])
        rows.append(("abl", r, t.dif_media, t.ic_bajo, t.ic_alto, k))
    df = pd.DataFrame(rows, columns=["grp", "m", "d", "lo", "hi", "k"])
    df = pd.concat([df[df.grp == "bench"].sort_values("d"), df[df.grp == "abl"].sort_values("d")])
    y = np.arange(len(df))[::-1].astype(float)
    y[df.grp.values == "abl"] -= 0.8
    for yi, (_, r) in zip(y, df.iterrows()):
        sig = r.k >= 3
        col = FOCAL if sig else GREY
        a.plot([r.lo * 1e3, r.hi * 1e3], [yi, yi], color=col, lw=2.2, solid_capstyle="butt")
        a.plot(r.d * 1e3, yi, "o", ms=7, mfc=col if sig else "white", mec=col, mew=1.8)
    a.axvline(0, color=DARK, lw=1.0)
    a.set_yticks(y, [NAMES[m] for m in df.m])
    a.set_xscale("symlog", linthresh=1.0)
    a.set_xticks([-30, -10, -3, -1, 0, 1])
    a.set_xticklabels(["−30", "−10", "−3", "−1", "0", "1"])
    a.set_xlabel("Δ log-loss vs. comparator (×10⁻³), Telco")
    a.grid(axis="y", visible=False)
    ab = y[df.grp.values == "abl"]
    a.text(1.02, ab.mean(), "Ablation", transform=a.get_yaxis_transform(), rotation=90, va="center", fontsize=12, color=DARK)
    bb = y[df.grp.values == "bench"]
    a.text(1.02, bb.mean(), "Benchmarks", transform=a.get_yaxis_transform(), rotation=90, va="center", fontsize=12, color=DARK)
    a.text(0.0, 1.01, "← RegionalMAPEM better", transform=a.transAxes, fontsize=12, color=DARK, va="bottom")
    a.plot([], [], "o", mfc=FOCAL, mec=FOCAL, label="≥ 3 tests reject (Holm)")
    a.plot([], [], "o", mfc="white", mec=GREY, mew=1.8, label="< 3 tests reject")
    a.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    a.margins(y=0.03)

    rows = []
    best = {"telco": "logistic_splines", "uci_563_iranian": "hist_gb", "openml_40701_churn": "catboost",
            "openml_1112_kddcup09_churn": "hist_gb"}
    for ds, r in best.items():
        t = t1_row(ds, r)
        blocks = ["A1_principal_benchmarks"] if ds == "telco" else ["D_adicionales"]
        rows.append((ds, r, t.dif_media, t.ic_bajo, t.ic_alto, n_rejections(ds, r, blocks)))
    yb = np.arange(len(rows))[::-1]
    for yi, (ds, r, d, lo, hi, k) in zip(yb, rows):
        col = FOCAL if k >= 1 else GREY
        b.plot([lo * 1e3, hi * 1e3], [yi, yi], color=col, lw=2.2, solid_capstyle="butt")
        b.plot(d * 1e3, yi, "o", ms=7, mfc=col if k >= 1 else "white", mec=col, mew=1.8)
    b.axvline(0, color=DARK, lw=1.0)
    b.set_yticks(yb, [f"{DS[ds]} vs. {NAMES[r]}" for ds, r, *_ in rows])
    b.set_xlabel("Δ log-loss vs. best benchmark (×10⁻³)")
    b.grid(axis="y", visible=False)
    b.margins(y=0.12, x=0.05)
    for ax, L in ((a, "a"), (b, "b")):
        ax.text(-0.02, 1.03, L, transform=ax.transAxes, fontweight="bold", fontsize=13, ha="right", va="bottom")
    fig.savefig(os.path.join(OUT, "fig_differences.pdf"))
    fig.savefig(os.path.join(OUT, "fig_differences.png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Figura 3: distribución por pliegue
def fig_boxplot():
    t = mpp[mpp.conjunto == "telco"]
    order = t.groupby("modelo").log_loss.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(W, 6.4))
    data = [t[t.modelo == m].log_loss.values for m in order][::-1]
    labels = [NAMES[m] for m in order][::-1]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color=DARK, lw=1.8))
    for patch, m in zip(bp["boxes"], order[::-1]):
        focal = m == "RegionalMAPEM"
        patch.set_facecolor("#9FB6D1" if focal else "#E6EAEF")
        patch.set_edgecolor(FOCAL if focal else GREY)
        patch.set_linewidth(2.0 if focal else 1.2)
    rng = np.random.default_rng(42)
    for i, d in enumerate(data, start=1):
        ax.plot(d, i + rng.uniform(-0.18, 0.18, len(d)), ".", color=DARK, ms=3, alpha=0.5)
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.set_xlabel("Log-loss per outer fold (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.03)
    fig.savefig(os.path.join(OUT, "fig_boxplot_folds.pdf"))
    fig.savefig(os.path.join(OUT, "fig_boxplot_folds.png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Figura 4: sensibilidad a K y c
def fig_sensitivity():
    s = pd.read_csv(os.path.join(ROOT, "experiments", "resultados", "robustez", "sensibilidad_K_c.csv"))
    ref = mpp[(mpp.conjunto == "telco") & (mpp.modelo == "logistic_splines")][["rep", "pliegue", "log_loss"]]
    s = s.merge(ref.rename(columns={"log_loss": "ref"}), on=["rep", "pliegue"], how="inner")
    s["d"] = (s.log_loss - s.ref) * 1e3
    piv = s.groupby(["K", "c"]).d.mean().unstack("c")
    nfold = s.groupby(["K", "c"]).size().min()
    fig, ax = plt.subplots(figsize=(W, 4.6))
    lim = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xlabel("Prior strength c")
    ax.set_ylabel("Number of regions K")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.6 * lim else DARK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Δ log-loss vs. spline logistic (×10⁻³)")
    fig.savefig(os.path.join(OUT, "fig_sensitivity_K_c.pdf"))
    fig.savefig(os.path.join(OUT, "fig_sensitivity_K_c.png"), dpi=200)
    plt.close(fig)
    return piv, nfold


# ------------------------------------------------------------------ Figura 5: calibración
def fig_calibration():
    c = pd.read_csv(os.path.join(TAB, "libro", "calibracion_telco.csv"))
    sel = [("RegionalMAPEM", FOCAL, "-", "o"), ("logistic_splines", "#E0A526", "--", "s"),
           ("xgboost", "#7A7A7A", ":", "^"), ("naive_bayes", "#B5651D", "-.", "D")]
    fig, ax = plt.subplots(figsize=(W, 4.8))
    ax.plot([0, 1], [0, 1], color=GREY, lw=1.2, label="Perfect calibration")
    for m, col, ls, mk in sel:
        d = c[(c.modelo == m) & (c.n >= 30)].sort_values("p_media")
        ax.plot(d.p_media, d.frecuencia_observada, ls=ls, marker=mk, color=col, ms=5, label=NAMES[m])
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig_calibration.pdf"))
    fig.savefig(os.path.join(OUT, "fig_calibration.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_differences()
    fig_boxplot()
    piv, nfold = fig_sensitivity()
    fig_calibration()
    print("folds per (K, c):", nfold)
    print(piv.round(2).to_string())
