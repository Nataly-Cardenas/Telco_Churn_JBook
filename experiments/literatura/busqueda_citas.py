"""Búsqueda por citación y por palabras clave en índices abiertos (OpenAlex y arXiv).

Complementa la búsqueda en WoS, Scopus y Google Académico. Para cada trabajo semilla descarga las citas
hacia atrás y hacia adelante en OpenAlex y conserva las que mencionan churn o abandono de clientes en el
título o el resumen. También corre cadenas de búsqueda con fecha. Guarda los registros en esta carpeta.
Uso: python experiments/literatura/busqueda_citas.py
"""
import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DEST = Path(__file__).resolve().parent
API = "https://api.openalex.org/works"
HOY = dt.date.today().isoformat()
SEMILLAS = {
    "De Caigny 2018 (logit leaf model)": "A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees",
    "Soares 2006 (DES-Clustering)": "Using Accuracy and Diversity to Select Classifiers to Build Ensembles",
    "Yan 2001 (MoE en churn)": "Improving prediction of customer behavior in nonstationary environments",
    "Sill 2009 (FWLS)": "Feature-Weighted Linear Stacking",
    "Jordan y Jacobs 1994 (MoE)": "Hierarchical mixtures of experts and the EM algorithm",
}
CADENAS = [
    '"mixture of experts" churn',
    '"feature-weighted linear stacking" churn',
    '"locally weighted stacking" churn',
    'stacking "local weights" churn',
    '"dynamic ensemble selection" churn',
    '"dynamic classifier selection" churn',
    '"Dirichlet prior" ensemble weights churn',
    '"expectation maximization" ensemble weights churn prediction',
    '"gaussian mixture" ensemble weights churn prediction',
    'cluster-wise ensemble weights churn prediction',
    '"region-based" ensemble churn prediction',
    'ponderación por regiones ensamble abandono de clientes',
]
ARXIV = [
    'all:"mixture of experts" AND all:churn',
    'all:"stacking" AND all:"local" AND all:churn',
    'all:"dynamic ensemble selection" AND all:churn',
    'all:"Dirichlet" AND all:ensemble AND all:churn',
    'all:"gaussian mixture" AND all:ensemble AND all:churn',
]
CHURN = re.compile(r"churn|attrition|customer retention|abandono", re.I)
MECANISMO = re.compile(r"mixture of experts|gating|local weight|locally weighted|region|cluster|segment|dynamic (ensemble|classifier|selection)|"
                       r"dirichlet|expectation.maximi[sz]ation|\bEM\b|gaussian mixture|feature-weighted", re.I)


def pedir(url):
    req = urllib.request.Request(url, headers={"User-Agent": "telco-churn-literature-review"})
    for intento in range(4):
        try:
            return urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
        except Exception:
            time.sleep(3 * (intento + 1))
    raise RuntimeError(url)


def get(params):
    return json.loads(pedir(API + "?" + urllib.parse.urlencode(params)))


def resumen(w):
    idx = w.get("abstract_inverted_index") or {}
    pos = sorted((p, k) for k, v in idx.items() for p in v)
    return " ".join(k for _, k in pos)


def fila(w, origen, sentido):
    return {"origen": origen, "sentido": sentido, "titulo": w.get("display_name"), "anio": w.get("publication_year"),
            "doi": w.get("doi"), "resumen": resumen(w)[:1200]}


def todas(filtro):
    salida, pagina = [], 1
    while pagina <= 25:
        d = get({"filter": filtro, "per-page": 100, "page": pagina,
                 "select": "id,display_name,publication_year,doi,abstract_inverted_index"})
        salida += d["results"]
        if len(salida) >= d["meta"]["count"] or not d["results"]:
            break
        pagina += 1
    return salida


def contar(filtro):
    return get({"filter": filtro, "per-page": 1, "select": "id"})["meta"]["count"]


snow, resumen_snow = [], []
for nombre, titulo in SEMILLAS.items():
    d = get({"search": titulo, "per-page": 5, "select": "id,display_name,publication_year,cited_by_count,referenced_works"})
    w = next((x for x in d["results"] if x["display_name"] and x["display_name"].lower()[:30] == titulo.lower()[:30]), None)
    if w is None:
        resumen_snow.append({"semilla": nombre, "encontrada": False})
        continue
    wid = w["id"].split("/")[-1]
    n_adelante = contar(f"cites:{wid}")
    adelante = todas(f"cites:{wid},title_and_abstract.search:churn OR attrition OR retention")
    atras = []
    refs = [r.split("/")[-1] for r in w.get("referenced_works", [])]
    for i in range(0, len(refs), 50):
        atras += todas("openalex:" + "|".join(refs[i:i + 50]))
    for sentido, lista in (("adelante", adelante), ("atras", atras)):
        for x in lista:
            texto = f"{x.get('display_name')} {resumen(x)}"
            if CHURN.search(texto):
                snow.append(fila(x, nombre, sentido) | {"menciona_mecanismo": bool(MECANISMO.search(texto))})
    resumen_snow.append({"semilla": nombre, "encontrada": True, "id_openalex": wid, "anio": w["publication_year"],
                         "citas_adelante": n_adelante, "referencias_atras": len(atras),
                         "con_churn": sum(1 for s in snow if s["origen"] == nombre)})
    print(resumen_snow[-1], flush=True)
pd.DataFrame(resumen_snow).assign(fecha=HOY).to_csv(DEST / "snowballing_resumen.csv", index=False, encoding="utf-8")
pd.DataFrame(snow).drop_duplicates(["origen", "titulo"]).to_csv(DEST / "snowballing_churn.csv", index=False, encoding="utf-8")

filas = []
for c in CADENAS:
    d = get({"search": c, "per-page": 25, "select": "id,display_name,publication_year,doi,abstract_inverted_index"})
    rev = d["results"]
    rel = [x for x in rev if CHURN.search(f"{x.get('display_name')} {resumen(x)}") and MECANISMO.search(f"{x.get('display_name')} {resumen(x)}")]
    filas.append({"base": "OpenAlex", "cadena": c, "resultados": d["meta"]["count"], "revisados": len(rev),
                  "candidatos": len(rel), "titulos_candidatos": " | ".join(str(x["display_name"]) for x in rel), "fecha": HOY})
for c in ARXIV:
    xml = pedir("http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": c, "max_results": 25}))
    total = int(re.search(r"<opensearch:totalResults[^>]*>(\d+)<", xml).group(1))
    titulos = [re.sub(r"\s+", " ", t) for t in re.findall(r"<entry>.*?<title>(.*?)</title>", xml, flags=re.S)]
    filas.append({"base": "arXiv", "cadena": c, "resultados": total, "revisados": len(titulos), "candidatos": len(titulos),
                  "titulos_candidatos": " | ".join(titulos), "fecha": HOY})
    time.sleep(3)
pd.DataFrame(filas).to_csv(DEST / "matriz_busqueda_abierta.csv", index=False, encoding="utf-8")
print(pd.DataFrame(filas)[["base", "cadena", "resultados", "revisados", "candidatos"]].to_string())
