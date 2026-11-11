"""Construye y ejecuta los notebooks de resultados del libro (capítulos 6, 7 y 10) a partir de las tablas versionadas.

Los notebooks solo leen experiments/resultados/tablas y no reentrenan nada. Se ejecutan aquí una vez y se guardan
con sus salidas, porque el libro se construye sin ejecutar notebooks.
Uso: python experiments/confirmatorio/construir_capitulos.py [06|07|10 ...]
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t):
    c = nbf.v4.new_code_cell(t.strip("\n"))
    c.metadata["tags"] = ["hide-input"]
    return c


def construir(nombre, celdas):
    nb = nbf.v4.new_notebook()
    nb.cells = celdas
    nb.metadata["kernelspec"] = {"display_name": "ml_project", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(RAIZ)}}).execute()
    for c in nb.cells:
        c.metadata.pop("execution", None)
    nbf.write(nb, RAIZ / nombre)
    print("escrito", nombre, flush=True)


if __name__ == "__main__":
    import importlib
    for cap in sys.argv[1:] or ["06", "07", "10"]:
        m = importlib.import_module(f"capitulo_{cap}")
        construir(m.ARCHIVO, m.CELDAS)
