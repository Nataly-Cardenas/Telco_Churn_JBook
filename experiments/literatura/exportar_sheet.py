"""Exporta a CSV las pestañas de la revisión de literatura del libro Candidatos_a_investigar.xlsx.

Solo cambia el formato: las fechas seriales de Excel pasan a AAAA-MM-DD y los enteros guardados como 8779.0 pasan a 8779.
Los textos y los conteos no se modifican. Se omite la fila de totales de la pestaña Busquedas, que es una fórmula.

Uso: python experiments/literatura/exportar_sheet.py ruta/a/Candidatos_a_investigar.xlsx
"""
import csv
import datetime as dt
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

M = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
SALIDA = Path(__file__).resolve().parent


def columna(ref):
    n = 0
    for c in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(c) - 64
    return n - 1


def leer(ruta):
    z = zipfile.ZipFile(ruta)
    cadenas = ["".join(t.text or "" for t in si.iter(f"{{{M}}}t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    destinos = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    hojas = {}
    for h in ET.fromstring(z.read("xl/workbook.xml")).find(f"{{{M}}}sheets"):
        d = destinos[h.get(f"{{{R}}}id")].lstrip("/")
        filas = []
        for f in ET.fromstring(z.read(d if d.startswith("xl/") else "xl/" + d)).iter(f"{{{M}}}row"):
            celdas = {}
            for c in f.findall(f"{{{M}}}c"):
                v = c.find(f"{{{M}}}v")
                celdas[columna(c.get("r"))] = cadenas[int(v.text)] if c.get("t") == "s" and v is not None else (v.text if v is not None else "")
            filas.append([celdas.get(i, "") for i in range(max(celdas) + 1)] if celdas else [])
        hojas[h.get("name")] = filas
    return hojas


def fecha(x):
    return (dt.date(1899, 12, 30) + dt.timedelta(days=int(float(x)))).isoformat() if re.fullmatch(r"\d{5}(\.0)?", x) else x


def entero(x):
    return x[:-2] if re.fullmatch(r"\d+\.0", x) else x


def tabla(filas, encabezado, nombre, col_fecha, cols_num):
    i = next(k for k, f in enumerate(filas) if f[:1] == [encabezado[0]] or f[:len(encabezado)] == encabezado)
    n = len(filas[i])
    salida = []
    for f in filas[i + 1:]:
        f = (f + [""] * n)[:n]
        if not any(x.strip() for x in f):
            continue
        if col_fecha is not None:
            f[col_fecha] = fecha(f[col_fecha])
        for c in cols_num:
            f[c] = entero(f[c])
        salida.append(f)
    with open(SALIDA / nombre, "w", newline="", encoding="utf-8") as a:
        w = csv.writer(a)
        w.writerow(filas[i][:n])
        w.writerows(salida)
    print(nombre, len(salida), "filas")


if __name__ == "__main__":
    H = leer(sys.argv[1])
    b = [f for f in H["Busquedas"] if not (len(f) > 5 and f[0] == "" and f[5] == "Total")]
    tabla(b, ["ID"], "registro_busquedas.csv", 5, [6, 7, 8])
    tabla(H["Matriz_novedad"], ["Fecha"], "matriz_novedad.csv", 0, [4, 5, 6])
    tabla(H["Diferenciacion"], ["Método afín"], "diferenciacion_sheet.csv", None, [])
