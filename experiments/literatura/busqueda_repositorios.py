"""Búsqueda de repositorios de código afines en GitHub, con la API de búsqueda y el cliente gh.

Salida: repositorios_cadenas.csv (cadena, total de resultados, fecha) y repositorios_resultados.csv (hasta 30 repositorios
por cadena, ordenados por relevancia; en las búsquedas de código, la descripción es la ruta del archivo). La clasificación manual de los cercanos va aparte, en repositorios_revision.csv.

Uso: python experiments/literatura/busqueda_repositorios.py
"""
import csv
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

SALIDA = Path(__file__).resolve().parent
CADENAS = [
    # familia afín + churn
    '"mixture of experts" churn', '"dynamic ensemble selection" churn', '"dynamic classifier selection" churn',
    'stacking "local weights" churn', '"weighted ensemble" churn', 'ensemble "expectation maximization" churn',
    'ensemble dirichlet churn', '"gaussian mixture" ensemble churn', 'cluster ensemble weights churn',
    'segment ensemble weights churn', '"abandono de clientes" ensamble',
    # el mecanismo, sin restringir a churn
    '"mixture of experts" "fixed experts"', '"regional" ensemble weights "expectation maximization"',
    '"local weights" ensemble classifiers "EM"', 'MAP-EM ensemble weights dirichlet prior', 'region-dependent ensemble weights classifiers',
    '"feature-weighted linear stacking"', '"linear pool" classifier probabilities',
]


# búsquedas en el contenido del código, para no depender de cómo se describe cada repositorio
CODIGO = [
    'churn "mixture of experts" language:Python', 'churn "gating" "experts" "predict_proba" language:Python',
    'churn "dynamic ensemble selection" language:Python', 'churn "KNORA" language:Python', 'churn deslib language:Python',
    'churn "GaussianMixture" "predict_proba" "weights" ensemble language:Python', 'churn dirichlet "predict_proba" ensemble weights language:Python',
    '"responsibilities" "predict_proba" "dirichlet" "experts" language:Python',
]


def consultar(q, tipo="repositories"):
    q = q + " in:name,description,readme" if tipo == "repositories" else q
    r = subprocess.run(["gh", "api", "-X", "GET", f"search/{tipo}", "-f", f"q={q}", "-f", "per_page=30"],
                       capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(r.stdout)


if __name__ == "__main__":
    hoy = dt.date.today().isoformat()
    cadenas, repos = [], []
    for q in CADENAS:
        d = consultar(q)
        cadenas.append([hoy, "GitHub", q, d["total_count"], len(d["items"])])
        repos += [[q, i["full_name"], i["html_url"], (i["description"] or "").replace("\n", " "), i["language"] or "", i["stargazers_count"],
                   i["updated_at"][:10]] for i in d["items"]]
        time.sleep(2.5)                                     # límite de la API de búsqueda: 30 consultas por minuto
    for q in CODIGO:
        d = consultar(q, "code")
        cadenas.append([hoy, "GitHub (código)", q, d["total_count"], len(d["items"])])
        repos += [[q, i["repository"]["full_name"], i["html_url"], i["path"], "", "", ""] for i in d["items"]]
        time.sleep(7)                                       # límite de la búsqueda de código: 10 consultas por minuto
    with open(SALIDA / "repositorios_cadenas.csv", "w", newline="", encoding="utf-8") as a:
        w = csv.writer(a); w.writerow(["fecha", "base", "cadena", "resultados", "revisados"]); w.writerows(cadenas)
    with open(SALIDA / "repositorios_resultados.csv", "w", newline="", encoding="utf-8") as a:
        w = csv.writer(a); w.writerow(["cadena", "repositorio", "url", "descripcion", "lenguaje", "estrellas", "actualizado"]); w.writerows(repos)
    for c in cadenas:
        print(c[3], "|", c[2])
    print("repositorios distintos:", len({r[1] for r in repos}))
