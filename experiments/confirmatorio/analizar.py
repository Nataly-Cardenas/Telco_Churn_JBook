"""Compila los resultados crudos y calcula la familia de pruebas de experiments/familia_holm.json.

Holm se aplica una sola vez sobre las 223 comparaciones. Si un conjunto todavía no terminó, sus pruebas
quedan sin calcular y el resultado se marca como parcial: en ese caso los p ajustados no son definitivos.
Uso: python experiments/confirmatorio/analizar.py
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import *  # noqa: E402,F403
import estadistica as E  # noqa: E402
from scipy import stats  # noqa: E402
from correr_anidado import ORIGINAL, ruta  # noqa: E402

SAL = RESULTADOS / "tablas"
SAL.mkdir(parents=True, exist_ok=True)
ABLACION = ["mapem_sin_prior", "mapem_K1", "mapem_regiones_duras", "promedio_simple", "stacking_logistico"]
MODELOS = BENCHMARKS + [ORIGINAL] + ABLACION
PERM = 100_000
CLAVE_METRICA = {"log_loss": "log_loss", "f1_positivo": "f1", "auc_roc": "auc"}


def cargar_pliegue(ds, r, f):
    d = {m: pickle.load(open(ruta(ds, r, f, m), "rb")) for m in BENCHMARKS}
    d.update(pickle.load(open(ruta(ds, r, f, "_combinadores"), "rb")))
    return d


def completo(ds):
    return all(ruta(ds, r, f, "_combinadores").exists() for r in range(REPETICIONES[ds]) for f in range(5))


def compilar(ds):
    """Tabla por pliegue y pérdida por cliente promediada sobre repeticiones."""
    _, y = cargar(ds)
    filas, costos = [], []
    perdidas = {m: np.zeros(len(y)) for m in MODELOS}
    for r in range(REPETICIONES[ds]):
        for f, (tr, te) in enumerate(pliegues_externos(y, r)):
            d = cargar_pliegue(ds, r, f)
            for m in MODELOS:
                p = np.clip(d[m]["pte"], EPS, 1 - EPS)
                filas.append({"conjunto": ds, "rep": r, "pliegue": f, "modelo": m, **metricas(y[te], p, d[m]["umbral"])})
                perdidas[m][te] += -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)) / REPETICIONES[ds]
                if "seg_busqueda" in d[m]:
                    costos.append({"conjunto": ds, "rep": r, "pliegue": f, "modelo": m, "ensayos": d[m]["ensayos"],
                                   "fallidos": d[m]["fallidos"], "seg_busqueda": d[m]["seg_busqueda"], "seg_ajuste": d[m]["seg_ajuste"],
                                   "seg_pred_por_1000": 1000 * d[m]["seg_pred"] / d[m]["n_pred"], "params": json.dumps(d[m]["params"], default=str)})
    return pd.DataFrame(filas), perdidas, pd.DataFrame(costos)


def serie(tab, modelo, metrica):
    return tab[tab.modelo == modelo].sort_values(["rep", "pliegue"])[metrica].to_numpy()


def signo(metrica):
    """Diferencias orientadas a pérdida: negativo significa que el original es mejor."""
    return 1.0 if metrica == "log_loss" else -1.0


def prueba(tab, perd, metrica, t, rival, semilla):
    k = CLAVE_METRICA[metrica]
    a, b = serie(tab, ORIGINAL, k), serie(tab, rival, k)
    d = signo(metrica) * (a - b)
    if t == "T1":
        est, p = E.nadeau_bengio(d)
    elif t == "T4":
        est, p = E.wilcoxon(d)
    else:
        est, p = E.permutacion_signos(perd[ORIGINAL] - perd[rival], PERM, semilla)
    return {"estadistico": est, "p_crudo": p, "dif_media": float(d.mean())}


def supuestos(d):
    d = np.asarray(d, float)
    sw = stats.shapiro(d).pvalue if len(d) >= 3 and d.std() > 0 else np.nan
    return {"shapiro_p": float(sw), "asimetria": float(stats.skew(d))}


def main():
    familia = json.loads((RAIZ / "experiments" / "familia_holm.json").read_text(encoding="utf-8"))["comparaciones"]
    listos = [d for d in DATASETS if completo(d)]
    print("conjuntos completos:", listos, flush=True)
    tabs, perds, costos = {}, {}, []
    for ds in listos:
        tabs[ds], perds[ds], c = compilar(ds)
        costos.append(c)
    pd.concat(tabs.values()).to_csv(SAL / "metricas_por_pliegue.csv", index=False)
    pd.concat(costos).to_csv(SAL / "costos_por_pliegue.csv", index=False)

    res = {}
    for ds in listos:
        M = tabs[ds].pivot_table(index=["rep", "pliegue"], columns="modelo", values="log_loss")
        univ = BENCHMARKS + [ORIGINAL] + ABLACION
        ff, pf, R, chi2 = E.friedman_iman_davenport(M[univ].to_numpy())
        res[ds] = {"M": M, "ff": ff, "pf": pf, "R": dict(zip(univ, R)), "chi2": chi2, "N": len(M), "k": len(univ)}
    if "telco" in tabs:
        rr = res["telco"]
        json.dump({"F_F": rr["ff"], "p": rr["pf"], "chi2": rr["chi2"], "N": rr["N"], "k": rr["k"], "rangos": rr["R"],
                   "CD": E.diferencia_critica(rr["k"], rr["N"])}, open(SAL / "friedman_telco.json", "w"), indent=1)

    # hold-out: se calcula solo si la corrida única ya existe
    ho = None
    if ruta("telco_holdout", 9, 0, "_combinadores").exists():
        _, yh = cargar_holdout()
        ho = {m: pickle.load(open(ruta("telco_holdout", 9, 0, m), "rb")) for m in BENCHMARKS}
        ho.update(pickle.load(open(ruta("telco_holdout", 9, 0, "_combinadores"), "rb")))
    mejor = {ds: min(BENCHMARKS, key=lambda m: tabs[ds][tabs[ds].modelo == m].log_loss.mean()) for ds in listos}

    # prueba entre conjuntos (rangos de log-loss medio por conjunto)
    entre = None
    if len(listos) == 4:
        univ = BENCHMARKS + [ORIGINAL]
        medias = np.array([[tabs[ds][tabs[ds].modelo == m].log_loss.mean() for m in univ] for ds in listos])
        ff, pf, R, chi2 = E.friedman_iman_davenport(medias)
        entre = {"medias": medias, "univ": univ, "ff": ff, "pf": pf, "R": R}
        pd.DataFrame(medias, index=listos, columns=univ).to_csv(SAL / "logloss_medio_por_conjunto.csv")

    filas = []
    for c in familia:
        fila = dict(c)
        ds, met, t, rival = c["dataset"], c["metrica"], c["prueba"], c["modelo_b"]
        sem = seed_for("test", c["id"])
        estad, p, dm = np.nan, np.nan, np.nan
        if c["bloque"] in ("A1_principal_benchmarks", "A2_principal_ablacion", "B_principal_secundarias", "D_adicionales") and ds in tabs:
            r = prueba(tabs[ds], perds[ds], met, t, rival, sem)
            estad, p, dm = r["estadistico"], r["p_crudo"], r["dif_media"]
        elif c["bloque"] == "A3_principal_friedman" and ds in res:
            rr = res[ds]
            if t == "T3_friedman":
                estad, p = rr["ff"], rr["pf"]
            else:
                dm = rr["R"][ORIGINAL] - rr["R"][rival]
                estad, p = dm, E.nemenyi_p(rr["R"][ORIGINAL], rr["R"][rival], rr["k"], rr["N"])
        elif c["bloque"] == "C_holdout" and ho is not None:
            rival_m = mejor["telco"]
            fila["modelo_b"] = rival_m
            po, pr = np.clip(ho[ORIGINAL]["pte"], EPS, 1 - EPS), np.clip(ho[rival_m]["pte"], EPS, 1 - EPS)
            if met == "log_loss":
                lo = -(yh * np.log(po) + (1 - yh) * np.log(1 - po))
                lr = -(yh * np.log(pr) + (1 - yh) * np.log(1 - pr))
                estad, p = E.permutacion_signos(lo - lr, PERM, sem)
                dm = estad
            elif met == "f1_positivo":
                ao, ar = (po >= ho[ORIGINAL]["umbral"]) == yh, (pr >= ho[rival_m]["umbral"]) == yh
                estad, p = E.mcnemar_exacto(ao, ar)
                dm = f1_score(yh, po >= ho[ORIGINAL]["umbral"]) - f1_score(yh, pr >= ho[rival_m]["umbral"])
            else:
                z, p, a1, a2 = E.delong(yh, po, pr)
                estad, dm = z, a1 - a2
        elif c["bloque"] == "E_entre_conjuntos" and entre is not None:
            if t == "friedman_iman_davenport":
                estad, p = entre["ff"], entre["pf"]
            else:
                io, ir = entre["univ"].index(ORIGINAL), entre["univ"].index(rival)
                d = entre["medias"][:, io] - entre["medias"][:, ir]
                estad, p = E.wilcoxon(d)
                dm = float(d.mean())
        fila.update(estadistico=estad, p_crudo=p, dif_media=dm)
        if c["bloque"] in ("A1_principal_benchmarks", "A2_principal_ablacion", "B_principal_secundarias", "D_adicionales") and ds in tabs:
            k = CLAVE_METRICA[met]
            a, b = serie(tabs[ds], ORIGINAL, k), serie(tabs[ds], rival, k)
            fila["cliff"] = E.cliff(a, b)
            fila["magnitud"] = E.magnitud_cliff(fila["cliff"])
            fila["pliegues_gana"] = int(np.sum(signo(met) * (a - b) < 0))
            fila["n_pliegues"] = len(a)
            fila.update(supuestos(a - b))
            if t == "T1":
                fila["ic_bajo"], fila["ic_alto"] = E.bca(signo(met) * (a - b), 10_000, sem)
        filas.append(fila)
    T = pd.DataFrame(filas)
    ok = T.p_crudo.notna()
    T["p_holm"] = np.nan
    T.loc[ok, "p_holm"] = E.holm(T.loc[ok, "p_crudo"].to_numpy())
    T["calculada"] = ok
    T.to_csv(SAL / "familia_resultados.csv", index=False, encoding="utf-8")
    parcial = int((~ok).sum())
    print(f"comparaciones calculadas: {ok.sum()} de {len(T)}" + (" (PARCIAL: Holm no definitivo)" if parcial else ""))

    # exploratorio: bayesiana de rangos con signo, ROPE de 0,001
    bay = []
    for ds in listos:
        for rival in BENCHMARKS + ABLACION:
            d = serie(tabs[ds], ORIGINAL, "log_loss") - serie(tabs[ds], rival, "log_loss")
            g, i, pe = E.bayes_rangos_signo(d, 0.001, 20_000, seed_for("test", 500, DATASETS.index(ds)))
            bay.append({"conjunto": ds, "rival": rival, "p_original_mejor": g, "p_equivalentes": i, "p_rival_mejor": pe})
    pd.DataFrame(bay).to_csv(SAL / "bayesiana_rope.csv", index=False)

    # resumen por modelo y veredicto R3
    resumen = []
    for ds in listos:
        for m in MODELOS:
            s = tabs[ds][tabs[ds].modelo == m]
            base = {"conjunto": ds, "modelo": m}
            for k in ("log_loss", "f1", "auc", "brier", "ece"):
                base[k + "_media"], base[k + "_de"] = s[k].mean(), s[k].std()
            lo, hi = E.bca(s.log_loss.to_numpy(), 10_000, seed_for("test", 600, MODELOS.index(m)))
            base["log_loss_ic_bajo"], base["log_loss_ic_alto"] = lo, hi
            resumen.append(base)
    pd.DataFrame(resumen).to_csv(SAL / "resumen_modelos.csv", index=False)

    ver = {"parcial": bool(parcial), "mejor_benchmark": mejor}
    if "telco" in tabs:
        mb = mejor["telco"]
        sel = T[(T.bloque == "A1_principal_benchmarks") & (T.modelo_b == mb)]
        nem = T[(T.prueba == "T3_nemenyi") & (T.modelo_b == mb)]
        media_o = tabs["telco"][tabs["telco"].modelo == ORIGINAL].log_loss.mean()
        media_b = tabs["telco"][tabs["telco"].modelo == mb].log_loss.mean()
        rechazos = {r.prueba: bool(r.p_holm < 0.05 and r.dif_media < 0) for r in sel.itertuples()}
        rechazos["T3_nemenyi"] = bool(nem.p_holm.iloc[0] < 0.05 and nem.dif_media.iloc[0] < 0)
        cl = float(sel.cliff.iloc[0])
        h2 = {}
        for ds in listos[1:]:
            mbd = mejor[ds]
            media = tabs[ds][tabs[ds].modelo == ORIGINAL].log_loss.mean()
            ref = tabs[ds][tabs[ds].modelo == mbd].log_loss.mean()
            sd = T[(T.bloque == "D_adicionales") & (T.dataset == ds) & (T.modelo_b == mbd)]
            h2[ds] = {"mejor_benchmark": mbd, "log_loss_original": media, "log_loss_mejor": ref, "mejor_en_media": bool(media < ref),
                      "significativamente_peor": bool(((sd.p_holm < 0.05) & (sd.dif_media > 0)).any())}
        cond1 = bool(media_o < media_b)
        cond2 = sum(rechazos.values()) >= 3
        cond3 = abs(cl) >= 0.147 and cl < 0
        cond4 = len(h2) == 3 and sum(v["mejor_en_media"] for v in h2.values()) >= 2 and not any(v["significativamente_peor"] for v in h2.values())
        ver.update(log_loss_original=media_o, log_loss_mejor_benchmark=media_b, diferencia=media_o - media_b, rechazos=rechazos, cliff=cl,
                   condicion_1=cond1, condicion_2=cond2, condicion_3=cond3, condicion_4=cond4, H2=h2, R3=bool(cond1 and cond2 and cond3 and cond4))
    (SAL / "veredicto_R3.json").write_text(json.dumps(ver, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(ver, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
