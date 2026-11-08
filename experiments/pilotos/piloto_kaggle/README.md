# Piloto exploratorio con el dataset del proyecto (2026-11-07)

**Este es el piloto válido.** Reemplaza al antiguo `piloto_logloss/` (archivado fuera del repo), que por error usó la versión extendida de IBM y los 7043 clientes (hold-out incluido).

## Diseño
- **Datos:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`, solo el conjunto de desarrollo (5634 clientes de `train_index.csv`). El hold-out (`test_index.csv`) no se usa.
- **Preprocesamiento:** el ETL por filas del proyecto (`HistoricalFeatures`) más `StandardScaler` y `OneHotEncoder(drop='first')`, ajustados dentro de cada pliegue.
- **Pliegues:** 5 repeticiones × 5 pliegues externos con `seed_for('outer', r)` (semilla raíz 42). La repetición 0 son los pliegues externos del entregable anterior. Los pesos se ajustan con cross-fitting interno de 3 pliegues (`seed_for('inner', r, f)`), así que no hay fuga.
- **Expertos:** 9 expertos (LR, LR con splines, RF, XGB, HistGB, kNN, CatBoost, LightGBM y Naive Bayes), sin SMOTE, calibrados con Platt (cv = 3) y con hiperparámetros fijos.
- **Modelo propio:** regiones con la regla del protocolo (variables continuas → cuantiles normales → PCA de 2 componentes → GMM propio), K = 4 y c = 20.

Scripts: `piloto_kaggle.py` guarda las predicciones de los expertos; `evaluar.py` aplica las reglas y corre las pruebas. Los resultados están en `resultados/`.

## Resultados (media ± desviación estándar en 25 pliegues)
| Modelo | log-loss | Brier | AUC | F1 |
|---|---|---|---|---|
| **RegionalMAPEM (K = 4, c = 20)** | **0,4116 ± 0,0087** | 0,1333 | 0,8500 | 0,634 |
| Pesos globales EM (K = 1) | 0,4117 | 0,1333 | 0,8500 | 0,633 |
| Stacking lineal | 0,4117 | 0,1334 | 0,8496 | 0,633 |
| LR con splines (mejor benchmark) | 0,4128 ± 0,0090 | 0,1335 | 0,8487 | 0,635 |
| HistGB | 0,4146 | 0,1345 | 0,8470 | 0,629 |
| LR | 0,4172 | 0,1352 | 0,8456 | 0,633 |
| LightGBM | 0,4172 | 0,1352 | 0,8445 | 0,627 |
| XGB | 0,4201 | 0,1350 | 0,8489 | 0,632 |
| RF | 0,4230 | 0,1360 | 0,8452 | 0,629 |
| CatBoost | 0,4246 | 0,1364 | 0,8455 | 0,627 |

## Lectura
- **Frente a los benchmarks:** con las tres pruebas (Nadeau-Bengio, Wilcoxon y permutación, corregidas con Holm), el modelo le gana a 7 de los 9 benchmarks.
- **Frente a HistGB:** gana en 21 de 25 pliegues y δ = −0,17, pero solo Wilcoxon es significativo.
- **Frente al mejor benchmark (LR con splines):** la diferencia es −0,0012, gana en 17 de 25 pliegues y solo Wilcoxon es significativo (p de Holm = 0,047). Nemenyi no detecta la diferencia y δ = −0,05 es despreciable. **R3 no se cumple.**
- **Ablación:** las regiones aportan 0,0001 frente a los pesos globales. El modelo empata con el stacking lineal, con la variante sin prior y con la de regiones duras.
- **Heterogeneidad:** el mejor experto cambia de una región a otra en 23 de 25 pliegues.
- **Convergencia:** el ascenso de la log-posterior fue monótono en todos los pliegues. Con 500 iteraciones (el valor del protocolo) no se alcanza la tolerancia de 1e-9, así que se recomienda subir a 5000 antes del tag.
- **F1:** todos los modelos calibrados quedan entre 0,62 y 0,635, por debajo del 0,638 del dashboard. Esas combinaciones optimizaron F1 con SMOTE, así que no son comparables en F1 con estos modelos optimizados para log-loss.
