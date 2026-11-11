"""Piloto exploratorio 8: ¿la fiabilidad relativa de los expertos cambia de verdad entre regiones?

Que "el mejor experto cambie entre regiones" no prueba heterogeneidad: con expertos casi empatados,
el ganador de cada región cambia por azar. Esta prueba de permutación separa la señal del azar.

Para cada uno de los 25 pliegues de desarrollo (caché de piloto_kaggle.py, sin hold-out):
  * T = ganancia de log-verosimilitud por cliente, dentro de la muestra (predicciones fuera de pliegue
    del pliegue de ajuste), de los pesos por región (K = 4, c = 0) frente a los pesos globales (K = 1).
  * H0: la fiabilidad relativa de los expertos no depende de la región. Bajo H0, permutar entre clientes
    las pertenencias a regiones (filas de R) no cambia la distribución de T. B = 200 permutaciones.
  * p = (1 + #{T_perm >= T}) / (B + 1).
También se cuenta, en los datos reales y bajo H0, con qué frecuencia cambia el mejor experto entre
regiones duras, para ver si el 23/25 del piloto 6b se distingue del azar.

Uso (desde la raíz del repo):
    python experiments/pilotos/piloto_heterogeneidad.py experiments/pilotos/piloto_kaggle/cache
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.original_model import EPS, RegionalMAPEM, map_em  # noqa: E402

B, K, MAX_ITER, TOL = 200, 4, 2000, 1e-9
OUT = Path(__file__).resolve().parent / "resultados_heterogeneidad"


def loglik(Lik, R, pi):
    return float(np.log(np.einsum("ik,km,im->i", R, pi, Lik)).mean())


def mejor_cambia(ll_inst, R):
    reg = R.argmax(1)
    return len({int(np.argmin(ll_inst[reg == k].mean(0))) for k in np.unique(reg)}) > 1


def un_pliegue(fp):
    d = np.load(fp)
    r, f = int(fp.split("_r")[1][0]), int(fp.split("_f")[-1][0])
    oof, Xtr, ytr = d["oof"], d["Xtr"], d["ytr"]
    seed = int(np.random.SeedSequence([42, 53, r, f]).generate_state(1)[0])
    m = RegionalMAPEM(n_regions=K, prior_strength=0.0, max_iter=MAX_ITER, tol=TOL,
                      random_state=seed).fit_from_predictions(Xtr, ytr, oof)
    R = m.train_regions_
    P = np.clip(oof, EPS, 1 - EPS)
    Lik = np.where(ytr[:, None] == 1, P, 1 - P)
    ll_inst = -np.log(Lik)
    M = P.shape[1]
    g, _, _ = map_em(Lik, np.ones((len(ytr), 1)), np.ones((1, M)), MAX_ITER, TOL)
    base = loglik(Lik, np.ones((len(ytr), 1)), g)
    T = loglik(Lik, R, m.weights_) - base
    rng = np.random.default_rng(seed)
    T_null, cambia_null = [], []
    for _ in range(B):
        Rp = R[rng.permutation(len(R))]
        pi_p, _, _ = map_em(Lik, Rp, np.ones((K, M)), MAX_ITER, TOL)
        T_null.append(loglik(Lik, Rp, pi_p) - base)
        cambia_null.append(mejor_cambia(ll_inst, Rp))
    T_null = np.array(T_null)
    return {"rep": r, "fold": f, "T": T, "T_null_media": T_null.mean(), "T_null_p95": np.quantile(T_null, 0.95),
            "p": (1 + np.sum(T_null >= T)) / (B + 1), "mejor_cambia": mejor_cambia(ll_inst, R),
            "frac_cambia_null": float(np.mean(cambia_null))}


def main():
    cache = Path(sys.argv[1])
    files = sorted(glob.glob(str(cache / "cache_r*_f*.npz")))
    assert len(files) == 25, f"se esperaban 25 pliegues en caché, hay {len(files)}"
    res = pd.DataFrame(Parallel(n_jobs=12)(delayed(un_pliegue)(fp) for fp in files))
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "permutacion_por_pliegue.csv", index=False)
    resumen = {
        "B": B, "K": K, "pliegues": len(res),
        "T_media": float(res["T"].mean()), "T_null_media": float(res["T_null_media"].mean()),
        "exceso_sobre_azar_media": float((res["T"] - res["T_null_media"]).mean()),
        "pliegues_p_menor_0_05": int((res["p"] < 0.05).sum()), "p_mediana": float(res["p"].median()),
        "p_rep0": res.loc[res.rep == 0, "p"].round(4).tolist(),
        "pliegues_mejor_cambia_real": int(res["mejor_cambia"].sum()),
        "frac_mejor_cambia_bajo_H0_media": float(res["frac_cambia_null"].mean()),
    }
    (OUT / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(res.round(5).to_string(index=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
