"""Descarga los 3 conjuntos adicionales del protocolo (sección 4.2) y los deja en experiments/datos/.

Fuentes públicas: OpenML 40701, UCI 563 y OpenML 1112. Imprime forma, tasa de churn y SHA-256; no ajusta
ningún modelo. Uso: python experiments/datos/descargar_datos.py
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

DEST = Path(__file__).resolve().parent


def resumen(nombre, df, objetivo):
    ruta = DEST / f"{nombre}.csv.gz"
    df.to_csv(ruta, index=False, compression={"method": "gzip", "mtime": 0})
    y = df[objetivo].astype(str)
    print(nombre, df.shape, y.value_counts(normalize=True).round(4).to_dict(),
          "faltantes", int(df.isna().sum().sum()), "duplicados", int(df.duplicated().sum()),
          hashlib.sha256(ruta.read_bytes()).hexdigest()[:16])


d = fetch_openml(data_id=40701, as_frame=True, parser="auto", data_home=str(DEST / "_cache_openml"))
resumen("openml_40701_churn", pd.concat([d.data, d.target.rename("class")], axis=1), "class")

url = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"
crudo = urllib.request.urlopen(url, timeout=120).read()
with zipfile.ZipFile(io.BytesIO(crudo)) as z:
    nombre = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    iran = pd.read_csv(z.open(nombre))
iran.columns = iran.columns.str.replace(r"\s+", " ", regex=True).str.strip()
resumen("uci_563_iranian", iran, "Churn")

d = fetch_openml(data_id=1112, as_frame=True, parser="auto", data_home=str(DEST / "_cache_openml"))
kdd = pd.concat([d.data, d.target.rename("class")], axis=1)
resumen("openml_1112_kddcup09_churn_completo", kdd, "class")
