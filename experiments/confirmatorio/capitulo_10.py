"""Capítulo 10 del libro: reproducibilidad y bibliografía."""
from construir_capitulos import code, md

ARCHIVO = "10_reproducibilidad.ipynb"

CELDAS = [
md(r"""
# Reproducibilidad y bibliografía

Todo lo que reporta este libro se puede reconstruir desde el repositorio. Este capítulo reúne lo necesario para hacerlo:
el entorno, las semillas, los espacios de búsqueda, la tabla maestra de experimentos, el tiempo de cómputo y la lista de
verificación. Los notebooks del libro no reentrenan modelos: leen resultados versionados en
`experiments/resultados/tablas/`, que a su vez salen de los scripts de `experiments/confirmatorio/`.
"""),
md(r"""
## Cómo reproducir los resultados

| Paso | Comando | Produce | Tiempo de reloj medido |
|---|---|---|---|
| 1. Validación anidada de los 4 conjuntos | `bash experiments/confirmatorio/correr_con_reintentos.sh 10` | `resultados/crudos/` (1020 checkpoints) | ≈ 31 h, incluida la caída de D1 |
| 2. Combinaciones del dashboard, robustez y tenure | `bash experiments/confirmatorio/correr_etapas_post.sh 10` | `tablas/combinaciones_dashboard.csv`, `resultados/robustez/`, `crudos/telco_tenure/` | 1 min, 53 min y 20 min |
| 3. Análisis confirmatorio y sello | `python experiments/confirmatorio/analizar.py` y `sellar_pre_holdout.py` | `tablas/*.csv`, `sello_pre_holdout.json` | 2,5 min |
| 4. Hold-out, análisis final, escalabilidad y SHAP | `bash experiments/confirmatorio/correr_etapas_post.sh 10 final` | `crudos/telco_holdout/`, `tablas/` completas | 18, 2,5, 4 y 24 min |
| 5. Tablas livianas del libro | `python experiments/confirmatorio/derivar_tablas_libro.py` (y `conjuntos`) | `tablas/libro/` | 5 min |
| 6. Capítulos 6, 7 y 10 | `python experiments/confirmatorio/construir_capitulos.py` | notebooks con sus salidas | 2 min |
| 7. Libro | `jupyter-book build .` | `_build/html` | 2 min |

Los checkpoints crudos (unos 115 MB) no se versionan; todo lo demás sí. Cada tarea guarda su checkpoint al terminar,
así que una corrida interrumpida se reanuda sin repetir trabajo. Con las semillas fijas, una corrida nueva produce los
mismos números: repetimos tres tareas al azar y coincidieron bit a bit con sus checkpoints (desviación D1).
"""),
code(r"""
# Entorno de software con el que se corrieron los experimentos
import json, sys, platform
from importlib.metadata import version
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from src.presentacion import tabla, nota, numero, NOMBRES, CONJUNTOS
TAB = ROOT / "experiments" / "resultados" / "tablas"
paquetes = ["numpy", "scipy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna", "imbalanced-learn",
            "shap", "joblib", "matplotlib"]
tabla(pd.DataFrame({"Componente": ["Python"] + paquetes, "Versión": [platform.python_version()] + [version(p) for p in paquetes]}),
      "Tabla 10.1. Versiones del software (entorno conda ml_project)", buscar=False,
      nota="requirements.txt fija estas mismas versiones; requirements-book.txt, las del libro (jupyter-book 0.15.1).")
"""),
md(r"""
**Hardware.** Portátil con Intel Core Ultra 5 125H (14 núcleos, 18 hilos lógicos, mezcla de núcleos de rendimiento y de
eficiencia), 7,5 GB de RAM utilizable, disco SSD de 512 GB y Windows 11 Home, sin GPU dedicada. Cada tarea corrió en un
proceso con un solo hilo por modelo, con 10 procesos en paralelo. Los tiempos de la tabla 6.1 dependen de esta máquina;
los resultados numéricos no.

## Semillas

Todas las semillas salen de una sola raíz, 42, combinada con el papel que cumple cada una y con sus coordenadas
(repetición, pliegue, modelo) mediante `numpy.random.SeedSequence`. Así ninguna semilla se reutiliza por accidente entre
papeles distintos, y cualquiera se puede recalcular.
"""),
code(r"""
# Derivación de semillas: seed_for(papel, *coordenadas) = SeedSequence([42, código del papel, *coordenadas])
sys.path.insert(0, str(ROOT / "experiments" / "confirmatorio"))
from comun import SEED_ROLES, seed_for
uso = {"subset": "Submuestra fija de KDDCup09", "outer": "Pliegues externos, por repetición", "inner": "Pliegues internos, por repetición y pliegue",
       "sampler": "Remuestreo (sin uso en este protocolo)", "model": "Semilla de cada modelo, por repetición, pliegue y modelo",
       "calibration": "Calibración (sin uso: Platt es determinista)", "test": "Permutaciones, bootstrap, SHAP y perturbaciones de robustez",
       "optimizer": "Muestreador TPE de Optuna, por repetición, pliegue y modelo"}
tabla(pd.DataFrame({"Papel": list(SEED_ROLES), "Código": list(SEED_ROLES.values()), "Uso": [uso[k] for k in SEED_ROLES],
                    "Ejemplo": [f"seed_for('{k}', 0, 0) = {seed_for(k, 0, 0)}" for k in SEED_ROLES]}),
      "Tabla 10.2. Papeles de las semillas", buscar=False)
"""),
md(r"""
## Espacios de búsqueda

Los rangos quedaron fijados en `experiments/espacios.json` antes del tag. Cada modelo arranca en sus valores
recomendados por la literatura o por su librería, que cuentan como la primera de las 30 evaluaciones.
"""),
code(r"""
# Rangos de hiperparámetros de cada modelo, con el punto de partida
ESP = json.loads((ROOT / "experiments" / "espacios.json").read_text(encoding="utf-8"))
filas = []
def rango(e):
    if e["tipo"] == "categorical":
        return " | ".join(map(str, e["valores"]))
    escala = " (log)" if e["tipo"] in ("log", "int_log") else ""
    return f"{numero(e['low'], 4).rstrip('0').rstrip(',')} a {numero(e['high'], 4).rstrip('0').rstrip(',')}{escala}"
for m, d in ESP["benchmarks"].items():
    for p, e in d["espacio"].items():
        filas.append({"Modelo": NOMBRES[m], "Hiperparámetro": p, "Rango": rango(e), "Inicio": str(d["inicio"].get(p, "—"))})
mo = ESP["modelo_original"]["RegionalMAPEM"]
for p, e in mo["espacio"].items():
    filas.append({"Modelo": "RegionalMAPEM", "Hiperparámetro": p, "Rango": rango(e), "Inicio": str(mo.get("inicio", {}).get(p, "—"))})
tabla(pd.DataFrame(filas), "Tabla 10.3. Espacios de búsqueda (30 evaluaciones de TPE por modelo y pliegue externo)", decimales=4)
"""),
md(r"""
## Tabla maestra de experimentos

La tabla maestra reúne, para cada conjunto y cada modelo, cuántas veces se ajustó, con qué presupuesto, cuánto costó y
qué resultado dio, como media y desviación estándar sobre los pliegues externos. Se guarda también como
`experiments/resultados/tablas/tabla_maestra_experimentos.csv`, que además incluye las 112 configuraciones del entregable
anterior (modelo, balanceo y optimizador) con sus métricas externas y tiempos; el detalle por pliegue y las semillas de cada
una están en `resultados_modelado/analisis_final/unidades.csv`.
"""),
code(r"""
# Tabla maestra: una fila por conjunto y modelo
POR_PLIEGUE = pd.read_csv(TAB / "metricas_por_pliegue.csv")
COSTOS = pd.read_csv(TAB / "costos_por_pliegue.csv")
ag = POR_PLIEGUE.groupby(["conjunto", "modelo"]).agg(pliegues=("pliegue", "size"), repeticiones=("rep", "nunique"),
        log_loss_media=("log_loss", "mean"), log_loss_de=("log_loss", "std"), f1_media=("f1", "mean"), f1_de=("f1", "std"),
        auc_media=("auc", "mean"), auc_de=("auc", "std"), brier_media=("brier", "mean"), ece_media=("ece", "mean")).reset_index()
co = COSTOS.groupby(["conjunto", "modelo"]).agg(evaluaciones_totales=("ensayos", "sum"), evaluaciones_fallidas=("fallidos", "sum"),
        horas_busqueda=("seg_busqueda", lambda s: s.sum() / 3600), seg_ajuste_medio=("seg_ajuste", "mean")).reset_index()
M = ag.merge(co, on=["conjunto", "modelo"], how="left")
M["pliegues_internos"] = 3; M["semilla_raiz"] = 42; M["calibracion"] = "Platt (CV = 3)"
# Archivo completo: este protocolo más las 112 configuraciones del entregable anterior (realización principal)
U = pd.read_csv(ROOT / "resultados_modelado" / "analisis_final" / "unidades.csv"); U = U[U.replicate == 0]
E3 = U.groupby(["model", "balance", "optimizer"]).agg(pliegues=("outer_fold", "size"), f1_media=("f1_positive", "mean"), f1_de=("f1_positive", "std"),
        auc_media=("roc_auc", "mean"), auc_de=("roc_auc", "std"), brier_media=("brier", "mean"), ece_media=("ece", "mean"), evaluaciones_totales=("evaluations", "sum"),
        horas_busqueda=("search_seconds", lambda s: s.sum() / 3600), seg_ajuste_medio=("refit_seconds", "mean")).reset_index().rename(columns={"model": "modelo", "balance": "balanceo", "optimizer": "optimizador"})
E3 = E3.assign(etapa="entregable anterior (112 configuraciones)", conjunto="telco", repeticiones=1, pliegues_internos=3, semilla_raiz=42, calibracion="sin calibrar")
pd.concat([M.assign(etapa="protocolo confirmatorio", balanceo="none", optimizador="bayesian_tpe"), E3], ignore_index=True).to_csv(TAB / "tabla_maestra_experimentos.csv", index=False)
vista = pd.DataFrame({"Conjunto": M.conjunto.map(CONJUNTOS), "Modelo": M.modelo.map(NOMBRES), "Pliegues externos": M.pliegues,
                      "Evaluaciones de hiperparámetros": M.evaluaciones_totales, "Fallidas": M.evaluaciones_fallidas,
                      "Horas de búsqueda": M.horas_busqueda,
                      "Log-loss": [f"{numero(a, 4)} ± {numero(b, 4)}" for a, b in zip(M.log_loss_media, M.log_loss_de)],
                      "F1": [f"{numero(a, 3)} ± {numero(b, 3)}" for a, b in zip(M.f1_media, M.f1_de)],
                      "AUC": [f"{numero(a, 3)} ± {numero(b, 3)}" for a, b in zip(M.auc_media, M.auc_de)]})
tabla(vista, f"Tabla 10.4. Tabla maestra de experimentos ({len(M)} filas)", decimales=2,
      nota="Las variantes sin búsqueda (promedio simple, pesos globales) no consumen evaluaciones. Las horas son de proceso, no de reloj.")
"""),
md(r"""
## Tiempo de cómputo
"""),
code(r"""
# Horas de proceso por conjunto (suma de búsqueda, ajuste y predicción de todos los modelos)
h = COSTOS.assign(h=(COSTOS.seg_busqueda + COSTOS.seg_ajuste + COSTOS.seg_pred_por_1000 * 0) / 3600).groupby("conjunto").h.sum()
reloj = {"telco": "2,4 h (más 1,1 h del intento con 16 procesos)", "uci_563_iranian": "3,5 h", "openml_40701_churn": "5,6 h", "openml_1112_kddcup09_churn": "19,6 h"}
tabla(pd.DataFrame({"Conjunto": [CONJUNTOS[c] for c in reloj], "Horas de proceso": [h[c] for c in reloj], "Tiempo de reloj con 10 procesos": list(reloj.values())}),
      "Tabla 10.5. Costo de la validación anidada", decimales=1, buscar=False)
"""),
md(r"""
KDDCup09 se llevó casi el 60 % del cómputo (166 de 281 horas de proceso) con solo 10 de los 85 pliegues externos: sus 908 columnas tras el
preprocesamiento encarecen sobre todo a Random Forest, CatBoost y HistGB.

## Lista de verificación de reproducibilidad

| Elemento | Dónde está |
|---|---|
| Versiones del software y del hardware | Tabla 10.1 y párrafo de hardware; `requirements.txt` |
| Semillas | Tabla 10.2; `seed_for` en `experiments/confirmatorio/comun.py` |
| Rangos de hiperparámetros | Tabla 10.3; `experiments/espacios.json` |
| Número de ejecuciones | 5 × 5 pliegues externos (2 × 5 en KDDCup09), 3 internos, 30 evaluaciones; tabla 10.4 |
| Medidas de dispersión | Desviación estándar e IC BCa en las tablas 6.2 y 10.4; semillas y repeticiones en la tabla 7.5 |
| Protocolo y desviaciones | `experiments/protocolo.md`, tag `protocolo-v1`, sección 12 |
| Familia de pruebas | `experiments/familia_holm.json` (223 comparaciones) |
| Datos | Telco: `WA_Fn-UseC_-Telco-Customer-Churn.csv` con `train_index.csv` y `test_index.csv`; adicionales en `experiments/datos/` |
| Código del modelo | `src/original_model.py`, con pruebas en `tests/test_original_model.py` |
| Código de los experimentos | `experiments/confirmatorio/` |
| Resultados | `experiments/resultados/tablas/` y `tablas_pre_holdout/` con su sello SHA-256 |
| Tiempo de cómputo | Tablas 6.1, 7.8 y 10.5 |
| Repositorio | [github.com/Nataly-Cardenas/Telco_Churn_JBook](https://github.com/Nataly-Cardenas/Telco_Churn_JBook), licencia MIT |

## Declaración de autoría y uso de herramientas

Laura S. Rivera Figueredo y Natalý A. Cárdenas Izaquita somos las autoras de este proyecto. Decidimos el problema,
la hipótesis, el modelo, el protocolo y las conclusiones, y respondemos por todo su contenido.

Durante el proyecto usamos Claude (Anthropic), a través de Claude Code, como asistente de programación y redacción. Lo
usamos para escribir y revisar código de los scripts de experimentación y análisis estadístico, la clase del modelo y
sus pruebas, para auditar tiempos de cómputo y la coherencia entre el protocolo y lo ejecutado, y para redactar
borradores de varios capítulos a partir de los resultados. Todas sus propuestas pasaron por nuestra revisión. Los
resultados numéricos salen de la ejecución del código del repositorio, no de la herramienta, y ninguna cifra del libro
se escribió sin verificarla contra las tablas versionadas. La herramienta no tuvo acceso al hold-out antes de su
evaluación única, que se hizo después de sellar el análisis confirmatorio.

Firmado por el equipo: **Laura S. Rivera Figueredo** y **Natalý A. Cárdenas Izaquita**.

## Referencias

```{bibliography}
:all:
```
"""),
]
