"""Evalúa reglas de combinación sobre las predicciones cacheadas (mismos 25 pliegues de desarrollo).
Las regiones siguen la regla del protocolo (sección 6) con las funciones propias de src/original_model.py."""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import logit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import original_model as om
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, f1_score
C = Path(sys.argv[1]); NOM = json.load(open(C / "expertos.json")); EPS = 1e-6
L = lambda p: np.clip(p, EPS, 1 - EPS)

def regiones(Xtr, Xte, K, seed, cov="full", hard=False, cols=None, n_pca=2):
    """Regla del protocolo: columnas continuas -> cuantiles normales -> PCA(2) -> GMM propio."""
    if K == 1: return np.ones((len(Xtr), 1)), np.ones((len(Xte), 1))
    cols = cols if cols is not None else [j for j in range(Xtr.shape[1]) if len(np.unique(Xtr[:, j])) > 2]
    sc = om._normal_scores_fit(Xtr[:, cols]); U = om._normal_scores(Xtr[:, cols], sc); V = om._normal_scores(Xte[:, cols], sc)
    mu, W = om._pca_fit(U, min(n_pca, U.shape[1])); Z, Zt = (U - mu) @ W.T, (V - mu) @ W.T
    m, s = Z.mean(0), Z.std(0) + 1e-12; Z, Zt = (Z - m) / s, (Zt - m) / s
    g = om._gmm_fit(Z, K, np.random.default_rng(seed), covariance=cov)
    R, Rt = om._gmm_resp(Z, *g), om._gmm_resp(Zt, *g)
    if hard: R, Rt = np.eye(K)[R.argmax(1)], np.eye(K)[Rt.argmax(1)]
    return R, Rt

def mapem(oof, ytr, Rtr, Rte, pte, c):
    Lk = np.where(ytr[:, None] == 1, L(oof), 1 - L(oof)); M = oof.shape[1]
    g = om.map_em(Lk, np.ones((len(ytr), 1)), np.ones((1, M)), 500, 1e-9)[0][0]
    pi, h, ok = om.map_em(Lk, Rtr, 1 + c * np.tile(g, (Rtr.shape[1], 1)), 500, 1e-9)
    ptr = np.einsum("ik,km,im->i", Rtr, pi, L(oof)); pt = np.einsum("ik,km,im->i", Rte, pi, L(pte))
    return ptr, pt, h, ok

def umbral(yv, p):
    grid = np.linspace(0.05, 0.95, 91); return grid[np.argmax([f1_score(yv, (p >= t).astype(int)) for t in grid])]

filas, perd, conv = [], {}, []
for fp in sorted(glob.glob(str(C / "cache_r*_f*.npz"))):
    d = np.load(fp); r, f = int(fp.split("_r")[1][0]), int(fp.split("_f")[-1][0]); fid = 5 * r + f
    oof, pte, Xtr, Xte, ytr, yte = d["oof"], d["pte"], d["Xtr"], d["Xte"], d["ytr"], d["yte"]
    P = {n: (pte[:, j], oof[:, j]) for j, n in enumerate(NOM)}
    Zo, Zt = logit(L(oof)), logit(L(pte)); stk = LogisticRegression(C=1.0, max_iter=3000).fit(Zo, ytr)
    P["Stacking lineal"] = (stk.predict_proba(Zt)[:, 1], stk.predict_proba(Zo)[:, 1])
    P["Promedio simple"] = (pte.mean(1), oof.mean(1))
    seed = om.np.random.SeedSequence([42, 53, r, f]).generate_state(1)[0]
    R1 = regiones(Xtr, Xte, 1, seed); R4 = regiones(Xtr, Xte, 4, seed); R4h = regiones(Xtr, Xte, 4, seed, hard=True)
    for nom, (Ra, Rb), c in [("Pesos globales EM (K=1)", R1, 0.0), ("RegionalMAPEM (K=4, c=20)", R4, 20.0),
                             ("Ablación: sin prior (c=0)", R4, 0.0), ("Ablación: regiones duras", R4h, 20.0)]:
        ptr, pt, h, ok = mapem(oof, ytr, Ra, Rb, pte, c); P[nom] = (pt, ptr)
        if nom.startswith("Regional"): conv.append(dict(pliegue=fid, iteraciones=len(h) - 1, convergio=ok, monotono=bool(np.all(np.diff(h) >= -1e-9))))
    for k, (pt, ptr) in P.items():
        pt = L(pt); thr = umbral(ytr, L(ptr))
        filas.append(dict(metodo=k, pliegue=fid, rep=r, log_loss=log_loss(yte, pt), brier=brier_score_loss(yte, pt),
                          auc=roc_auc_score(yte, pt), f1=f1_score(yte, (pt >= thr).astype(int))))
        v = perd.setdefault((k, r), {}); v.update(dict(zip(d["te"], -(yte * np.log(pt) + (1 - yte) * np.log(1 - pt)))))
    print(fid, flush=True)

res = pd.DataFrame(filas); res.to_csv(C / "metricas_por_pliegue.csv", index=False); pd.DataFrame(conv).to_csv(C / "convergencia.csv", index=False)
piv = res.pivot(index="pliegue", columns="metodo", values="log_loss"); J = len(piv)
best = piv[NOM].mean().idxmin(); PROPIO = "RegionalMAPEM (K=4, c=20)"
resumen = res.groupby("metodo")[["log_loss", "brier", "auc", "f1"]].agg(["mean", "std"]).sort_values(("log_loss", "mean")); resumen.to_csv(C / "resumen.csv")
k = piv.shape[1]; ranks = piv.rank(axis=1).mean()
CD = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * J))
fr = stats.friedmanchisquare(*[piv[c] for c in piv])
n = len(next(iter(perd.values())))
def inst(m, rep): v = perd[(m, rep)]; return np.array([v[i] for i in sorted(v)])
rows = []
for rival in [c for c in piv.columns if c != PROPIO]:
    dd = (piv[PROPIO] - piv[rival]).to_numpy()
    t = dd.mean() / np.sqrt(dd.var(ddof=1) * (1 / J + 0.25)); pnb = 2 * stats.t.sf(abs(t), J - 1)
    pw = stats.wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
    di = np.mean([inst(PROPIO, rr) - inst(rival, rr) for rr in range(5)], 0); rng = np.random.default_rng(0)
    sims = np.array([(di * rng.choice([-1, 1], len(di))).mean() for _ in range(10000)])
    pp = (np.sum(np.abs(sims) >= abs(di.mean())) + 1) / 10001
    rows.append(dict(rival=rival, dif_logloss=dd.mean(), pliegues_gana=int((dd < 0).sum()), p_NB=pnb, p_wilcoxon=pw, p_perm=pp,
                     nemenyi_sig=bool(abs(ranks[rival] - ranks[PROPIO]) > CD), cliff=np.sign(piv[PROPIO].values[:, None] - piv[rival].values[None]).mean(),
                     dz=-dd.mean() / dd.std(ddof=1)))
T = pd.DataFrame(rows); pv = T[["p_NB", "p_wilcoxon", "p_perm"]].to_numpy().ravel(); o = np.argsort(pv); adj = np.empty_like(pv); acc = 0
for i, j in enumerate(o): acc = max(acc, min(1, (len(pv) - i) * pv[j])); adj[j] = acc
T[["h_NB", "h_wilcoxon", "h_perm"]] = adj.reshape(-1, 3); T.to_csv(C / "pruebas_vs_rivales.csv", index=False)
# heterogeneidad: ¿cambia el mejor experto entre regiones? (OOF de entrenamiento)
het = []
for fp in sorted(glob.glob(str(C / "cache_r*_f*.npz"))):
    d = np.load(fp); R, _ = regiones(d["Xtr"], d["Xte"], 4, 0); reg = R.argmax(1); P = L(d["oof"]); yt = d["ytr"]
    ll = -(yt[:, None] * np.log(P) + (1 - yt[:, None]) * np.log(1 - P))
    het.append(len({NOM[int(np.argmin(ll[reg == kk].mean(0)))] for kk in range(4) if (reg == kk).sum() > 0}) > 1)
info = dict(dataset="WA_Fn-UseC_-Telco-Customer-Churn.csv (desarrollo, n=5634; hold-out sin usar)", mejor_benchmark=best,
            friedman_p=float(fr.pvalue), nemenyi_CD=float(CD), pliegues_mejor_experto_cambia=int(sum(het)), n_pliegues=len(het),
            convergio_todos=bool(pd.DataFrame(conv).convergio.all()), monotono_todos=bool(pd.DataFrame(conv).monotono.all()),
            iter_media=float(pd.DataFrame(conv).iteraciones.mean()))
json.dump(info, open(C / "info.json", "w"), indent=2, ensure_ascii=False)
pd.set_option("display.width", 250)
print(resumen.round(4).to_string()); print(T.round(4).to_string()); print(ranks.sort_values().round(2).to_string()); print(info)
