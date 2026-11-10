"""Presentación compartida de los capítulos de resultados: tablas, notas y estilo de figuras.

Reproduce el formato de las tablas del EDA (clase `eda-table` de style/componentes.css): título, buscador,
orden por columna y números con formato en español. No modifica los valores de origen.
"""
from html import escape

import matplotlib as mpl
import numpy as np
import pandas as pd
from IPython.display import HTML, display

AZUL, ACENTO, RIESGO, GRIS = "#1B4F72", "#06B6D4", "#E4572E", "#8A9BA8"

NOMBRES = {
    "knn": "kNN", "naive_bayes": "Naive Bayes", "logistic": "Logística", "decision_tree": "Árbol de decisión",
    "random_forest": "Random Forest", "xgboost": "XGBoost", "svm": "SVM", "logistic_splines": "Logística con splines",
    "hist_gb": "HistGB", "lightgbm": "LightGBM", "catboost": "CatBoost", "RegionalMAPEM": "RegionalMAPEM",
    "mapem_sin_prior": "MAP-EM sin prior", "mapem_K1": "Pesos globales (K = 1)",
    "mapem_regiones_duras": "Regiones duras", "promedio_simple": "Promedio simple",
    "stacking_logistico": "Stacking logístico",
}
CONJUNTOS = {"telco": "Telco", "uci_563_iranian": "Iranian (UCI 563)", "openml_40701_churn": "Churn (OpenML 40701)",
             "openml_1112_kddcup09_churn": "KDDCup09 (OpenML 1112)", "4_conjuntos": "Los 4 conjuntos",
             "telco_holdout": "Telco, hold-out"}

_contador = 0
_SCRIPT = """<script>(function(){const root=document.getElementById('__ID__');
const body=root.querySelector('tbody'),input=root.querySelector('input');
input.addEventListener('input',()=>{for(const r of body.rows)r.hidden=!r.textContent.toLowerCase().includes(input.value.toLowerCase());});
root.querySelectorAll('thead th').forEach((h,i)=>{let asc=true;h.tabIndex=0;h.title='Ordenar columna';
const num=t=>Number(t.replaceAll('.','').replace(',','.').replace('−','-').split(' ')[0]);
const sort=()=>{const rows=Array.from(body.rows);rows.sort((a,b)=>{const x=a.cells[i].textContent,y=b.cells[i].textContent;
const nx=num(x),ny=num(y);const c=(Number.isFinite(nx)&&Number.isFinite(ny))?nx-ny:x.localeCompare(y,'es');return asc?c:-c;});
rows.forEach(r=>body.appendChild(r));asc=!asc;};h.addEventListener('click',sort);h.addEventListener('keydown',e=>{if(e.key==='Enter')sort();});});})();</script>"""


def numero(x, decimales=4):
    """Número con coma decimal; notación científica para valores muy pequeños."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, (np.floating,)) and np.isnan(x)):
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "Sí" if x else "No"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", ".")
    if isinstance(x, (float, np.floating)):
        if x != 0 and abs(x) < 10 ** (-decimales):
            return f"{x:.2e}".replace(".", ",").replace("-", "−")
        s = f"{x:,.{decimales}f}".replace(",", "§").replace(".", ",").replace("§", ".")
        return s.replace("-", "−")
    return str(x)


def valor_p(p):
    if p is None or pd.isna(p):
        return "—"
    if p == 0:
        return "< 1e−16"
    return numero(p, 4) if p >= 0.001 else f"{p:.2e}".replace(".", ",").replace("-", "−")


def tabla(datos, titulo, decimales=4, columnas_p=(), nota=None, buscar=True):
    """Muestra un DataFrame con el formato de las tablas del libro (sin índice)."""
    display(HTML(tabla_html(datos, titulo, decimales, columnas_p, nota, buscar)))


def tabla_html(datos, titulo, decimales=4, columnas_p=(), nota=None, buscar=True):
    """HTML de una tabla con el formato del libro."""
    global _contador
    _contador += 1
    ident = f"tabla-res-{_contador}"
    t = datos.copy()
    t.columns.name = None
    vista = pd.DataFrame({c: [valor_p(v) if c in columnas_p else numero(v, decimales) for v in t[c]] for c in t.columns})
    # celdas cortas (números, intervalos) sin salto de línea; los textos largos sí pueden partirse
    celda = lambda v: f'<td style="white-space:nowrap !important">{escape(v)}</td>' if len(v) <= 26 else f'<td style="min-width:190px">{escape(v)}</td>'
    cabecera = "".join(f"<th>{escape(str(c))}</th>" for c in vista.columns)
    cuerpo = "".join("<tr>" + "".join(celda(v) for v in fila) + "</tr>" for fila in vista.itertuples(index=False))
    html = f'<table class="dataframe"><thead><tr>{cabecera}</tr></thead><tbody>{cuerpo}</tbody></table>'
    entrada = f'<input aria-label="Buscar en {escape(titulo)}" placeholder="Buscar en esta tabla…">' if buscar else '<input hidden>'
    pie = f'<span class="eda-table-note">{escape(nota)}</span>' if nota else ""
    return (f'<div class="eda-table" id="{ident}"><span class="eda-table-title">{escape(titulo)}</span>{entrada}{html}{pie}</div>'
            + _SCRIPT.replace("__ID__", ident))


def nota(texto_html, tipo=""):
    """Recuadro de lectura (tc-note); tipo '' (informativo), 'ok' o 'warn'."""
    display(HTML(f'<div class="tc-note {tipo}">{texto_html}</div>'))


def estilo_figuras():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlecolor": AZUL, "axes.labelcolor": "#243B53", "axes.edgecolor": "#9FB3C8", "axes.grid": True,
        "grid.color": "#E4EBF0", "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=[AZUL, RIESGO, ACENTO, "#F4B183", "#5B6B7C", "#2E86AB", "#9B59B6"]),
    })


def media_de(serie, decimales=4):
    return f"{numero(serie.mean(), decimales)} ± {numero(serie.std(), decimales)}"
