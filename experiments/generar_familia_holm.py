"""Genera experiments/familia_holm.json: la lista exhaustiva de comparaciones de la familia de Holm.

Toda prueba de hipótesis que se calcule en el experimento confirmatorio debe estar en esta lista
(y solo esas). Holm-Bonferroni se aplica una vez sobre el conjunto completo, con alfa = 0.05.
Uso: python experiments/generar_familia_holm.py
"""
import json
from pathlib import Path

BENCHMARKS = ["knn", "naive_bayes", "logistic", "decision_tree", "random_forest", "xgboost", "svm",
              "logistic_splines", "hist_gb", "lightgbm", "catboost"]
ABLACION = ["mapem_sin_prior", "mapem_K1", "mapem_regiones_duras", "promedio_simple",
            "stacking_logistico"]
ORIGINAL = "RegionalMAPEM"
ADICIONALES = ["openml_40701_churn", "uci_563_iranian", "openml_1112_kddcup09_churn"]
PRUEBAS_PAREADAS_LOGLOSS = {
    "T1": "t corregida de Nadeau-Bengio sobre las diferencias por pliegue (25; 10 en KDDCup09)",
    "T2": "permutación pareada de signos sobre pérdidas por instancia (100 000 permutaciones)",
    "T4": "Wilcoxon de rangos con signo sobre las diferencias por pliegue (25; 10 en KDDCup09)",
}
PRUEBAS_PAREADAS_SECUNDARIAS = {
    "T1": PRUEBAS_PAREADAS_LOGLOSS["T1"],
    "T4": PRUEBAS_PAREADAS_LOGLOSS["T4"],
}

comparaciones = []


def add(bloque, dataset, metrica, prueba, a, b, descripcion):
    comparaciones.append({"id": len(comparaciones) + 1, "bloque": bloque, "dataset": dataset,
                          "metrica": metrica, "prueba": prueba, "modelo_a": a, "modelo_b": b,
                          "descripcion": descripcion})


# A. Problema principal, log-loss: original frente a cada benchmark y cada variante de ablación
for rival in BENCHMARKS + ABLACION:
    bloque = "A1_principal_benchmarks" if rival in BENCHMARKS else "A2_principal_ablacion"
    for t, desc in PRUEBAS_PAREADAS_LOGLOSS.items():
        add(bloque, "telco", "log_loss", t, ORIGINAL, rival, desc)
add("A3_principal_friedman", "telco", "log_loss", "T3_friedman", "todos", "todos",
    "Friedman con corrección de Iman-Davenport sobre 17 modelos x 25 pliegues")
for rival in BENCHMARKS + ABLACION:
    add("A3_principal_friedman", "telco", "log_loss", "T3_nemenyi", ORIGINAL, rival,
        "comparación post hoc de Nemenyi (p del par)")

# B. Problema principal, métricas secundarias (F1 con umbral interno y AUC)
for metrica in ("f1_positivo", "auc_roc"):
    for rival in BENCHMARKS:
        for t, desc in PRUEBAS_PAREADAS_SECUNDARIAS.items():
            add("B_principal_secundarias", "telco", metrica, t, ORIGINAL, rival, desc)

# C. Hold-out (evaluación única) frente al mejor benchmark según el bucle externo
add("C_holdout", "telco_holdout", "log_loss", "permutacion", ORIGINAL, "mejor_benchmark",
    "permutación pareada sobre pérdidas por cliente del hold-out (1409)")
add("C_holdout", "telco_holdout", "f1_positivo", "mcnemar", ORIGINAL, "mejor_benchmark",
    "McNemar exacto sobre aciertos con el umbral del bucle interno")
add("C_holdout", "telco_holdout", "auc_roc", "delong", ORIGINAL, "mejor_benchmark",
    "DeLong para AUC correlacionadas (válido: observaciones independientes del hold-out)")

# D. Conjuntos adicionales, log-loss: original frente a cada benchmark
for ds in ADICIONALES:
    for rival in BENCHMARKS:
        for t, desc in PRUEBAS_PAREADAS_LOGLOSS.items():
            add("D_adicionales", ds, "log_loss", t, ORIGINAL, rival, desc)

# E. Entre conjuntos de datos (4 conjuntos)
add("E_entre_conjuntos", "4_conjuntos", "rango_log_loss", "friedman_iman_davenport", "todos", "todos",
    "Friedman con Iman-Davenport sobre rangos por conjunto (12 modelos x 4 conjuntos)")
for rival in BENCHMARKS:
    add("E_entre_conjuntos", "4_conjuntos", "log_loss", "wilcoxon", ORIGINAL, rival,
        "Wilcoxon por pares entre conjuntos (n = 4; potencia muy baja, se reporta)")

conteo = {}
for c in comparaciones:
    conteo[c["bloque"]] = conteo.get(c["bloque"], 0) + 1
salida = {
    "version": "protocolo-v1",
    "correccion": "Holm-Bonferroni sobre la familia completa, alfa = 0.05; se reportan p crudo y ajustado",
    "total_comparaciones": len(comparaciones),
    "umbral_minimo_holm": 0.05 / len(comparaciones),
    "conteo_por_bloque": conteo,
    "no_incluidas": [
        "prueba bayesiana de rangos con signo (ROPE ±0.001): exploratoria, no produce p-valor",
        "estudios de robustez de la sección 10 del protocolo: exploratorios",
    ],
    "nota_resolucion": "con 100 000 permutaciones el p mínimo de T2 es ~1e-5, por debajo del umbral mínimo de Holm",
    "comparaciones": comparaciones,
}
ruta = Path(__file__).resolve().parent / "familia_holm.json"
ruta.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(comparaciones), conteo, f"umbral mínimo {salida['umbral_minimo_holm']:.2e}")
