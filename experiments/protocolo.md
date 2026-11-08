# Protocolo experimental pre-registrado — Modelo original (Telco Churn)

> **Estado: BORRADOR, todavía sin congelar.** Este documento se congela con la etiqueta de Git
> `protocolo-v1` antes de ejecutar cualquier experimento confirmatorio. Cuando eso ocurra, la fecha
> y el hash del commit se anotan en la sección 13. Cualquier cambio posterior se declara como
> desviación (sección 12).
>
> Autoras: Nataly Cárdenas y Laura Rivera · Curso de Machine Learning, Universidad del Norte
> Revisión del 2026-11-07 (Laura): datos del piloto corregidos al dataset del proyecto, exposición previa del hold-out declarada, `max_iter` del MAP-EM a 5000 y pendientes de cómputo. Todo antes del tag.
>
> Guía de referencia: *Proyecto Final — Dashboard Predictivo, Modelo Original y Evaluación Rigurosa*,
> secciones 5 y 6, y criterios enviados por el docente por correo (R1, R3 y resultado negativo).

## 1. Objetivo y modelo evaluado

Se evalúa **RegionalMAPEM**: una ponderación por regiones del espacio de entrada que combina
expertos calibrados. Las regiones salen de una mezcla gaussiana programada por nosotras. Dentro de
cada región, los pesos de los expertos se estiman por MAP-EM con un prior de Dirichlet que los encoge
hacia los pesos globales:

$$
\hat p(\text{churn}\mid x)=\sum_{k=1}^{K} r_k(x)\sum_{m=1}^{M}\pi_{km}\,p_m(x),
\qquad \pi_{k\cdot}\sim\operatorname{Dirichlet}(\mathbf 1 + c\,\mathbf g),
$$

donde $r_k(x)$ es la pertenencia a la región $k$, $p_m(x)$ es la probabilidad calibrada del experto
$m$, $\mathbf g$ son los pesos globales (estimados con EM y $K=1$) y $c\ge 0$ es la fuerza del prior.
La formulación, la derivación de los pasos E y M y la demostración de convergencia están en el
capítulo 5 del libro.

## 2. Hipótesis

### 2.1 Confirmatorias

- **H1 (principal).** El log-loss medio de RegionalMAPEM en el bucle externo es distinto del de cada
  benchmark de la sección 5. La superioridad se declara solo si se cumplen las cuatro condiciones de
  la sección 9.
- **H2.** En los conjuntos de datos adicionales, RegionalMAPEM no es significativamente peor que el
  mejor benchmark en ninguno y es mejor en log-loss medio en al menos dos de los tres.

### 2.2 Exploratorias (no sustentan R3)

- La ablación (sección 6) y la contribución de cada componente.
- La equivalencia práctica mediante la prueba bayesiana de rangos con signo, con una ROPE de
  ±0,001 en log-loss.
- Los estudios de robustez de la Tabla 4 de la guía: sensibilidad a K y c, semillas, ruido, tamaño de
  muestra, cambio de distribución, calibración, predicción conforme y análisis de errores.
- SHAP del modelo original frente al mejor benchmark.

### 2.3 Expectativa declarada antes de los experimentos

Los pilotos de la sección 11 indican que **H1 probablemente no se cumplirá**. Con el dataset del proyecto
y solo el conjunto de desarrollo, la mejor combinación honesta mejoró entre 0,0007 (piloto 6) y 0,0012
(piloto 6b) de log-loss al mejor experto, con $|\delta| = 0{,}05$ en el piloto 6b, frente a unos 0,0024
necesarios para $|\delta|\ge 0{,}147$. Esto se declara aquí para que un resultado negativo no se interprete como
hallazgo inesperado, ni uno positivo como resultado buscado a posteriori.

## 3. Métricas

- **Principal: log-loss** (entropía cruzada binaria), con las probabilidades recortadas a
  $[10^{-6},\,1-10^{-6}]$.
  - **Cuándo se eligió.** Esta métrica **se eligió después de ver los pilotos**. Lo declaramos
    explícitamente, como pidió el docente.
  - **Por qué.** La justificación es sustantiva: en churn, la decisión de a quién intervenir y con qué
    presupuesto depende de probabilidades bien calibradas, no solo del orden de los clientes ni de un
    umbral. El log-loss es una regla de puntuación estrictamente propia y castiga la mala calibración.
  - **Lo que no cambia.** No se cambiará de métrica después del tag, aunque el resultado no sea
    significativo.
- **Secundarias:** F1 de la clase positiva, AUC-ROC, Brier y ECE con 10 bins de igual ancho. El F1 usa
  un umbral elegido dentro del bucle interno, maximizando el F1 sobre las predicciones fuera de
  pliegue. El procedimiento es el mismo para todos los modelos.

## 4. Datos y particiones

### 4.1 Problema principal

- **Datos:** IBM Telco Customer Churn, versión de Kaggle (7043 × 21), con el mismo ETL por filas del
  entregable 3 (`HistoricalFeatures`).
- **Conjunto de desarrollo:** 5634 clientes (`train_index.csv`). **Hold-out:** 1409 clientes
  (`test_index.csv`). El hold-out no se usa en ninguna decisión de diseño y se evalúa una sola vez al
  final, para el estudio de cambio de distribución y la validación externa.
- **Exposición previa del hold-out (declarada antes del tag).** Los pilotos 3–5 de la sección 11 se
  corrieron por error con la versión extendida de IBM y con los 7043 clientes, es decir, con el hold-out
  incluido. Sus cifras se descartaron, pero contribuyeron a elegir la familia del modelo. Por eso la
  evaluación final sobre el hold-out se reportará con esta advertencia. Desde el piloto 6b, ningún análisis
  usa el hold-out.
- **Validación anidada repetida:** 5 repeticiones × 5 pliegues externos estratificados.
  - La semilla externa de la repetición $r$ es `seed_for('outer', r)`, con $r=0,\dots,4$. La
    repetición 0 reproduce los pliegues externos del entregable 3.
  - El bucle interno tiene 3 pliegues estratificados, con `seed_for('inner', r, fold)`.
  - La semilla raíz es 42 y la derivación de semillas es la del entregable 3.
- **Pareado estricto:** todos los modelos, incluido el original, usan exactamente los mismos pliegues,
  semillas y preprocesamiento.

### 4.2 Conjuntos de datos adicionales

Se eligieron sin conocer el desempeño de ningún modelo en ellos. El criterio fue: repositorio público,
tarea de churn binaria y diferencias en tamaño, dimensión, desbalance y faltantes.

| Conjunto | Fuente | n | p | % churn | Particularidad |
|---|---|---|---|---|---|
| churn (telecom EE. UU.) | OpenML 40701, licencia "public" | 5000 | 20 | 14,1 | Variables de uso numéricas; procedencia original sin confirmar (PMLB/BigML) |
| Iranian Churn | UCI 563, licencia CC BY 4.0 | 3150 | 13 | 15,7 | Agregados de uso de 9 meses; otro país y otro operador |
| KDDCup09_churn (Orange) | OpenML 1112, licencia "Public" | 50 000 → submuestra estratificada fija de 10 000 | 230 | 7,3 | Alta dimensión, muchos faltantes, desbalance fuerte |

- **Submuestra de KDDCup09.** Se toma una submuestra estratificada de 10 000 filas con
  `seed_for('subset', 1112)` por razones de cómputo. La decisión se declara aquí, antes de ver
  resultados.
- **Mismo procedimiento.** En cada conjunto se aplica la misma validación anidada (5 × 5 externos y
  3 internos), los mismos modelos y el mismo presupuesto.
- **Preprocesamiento.** Se ajusta dentro de cada pliegue: imputación por mediana o moda, escalado y
  one-hot. Además, en KDDCup09 se eliminan las variables con más del 95 % de faltantes y las
  categóricas con más de 50 niveles se agrupan, con reglas aprendidas dentro del pliegue.

## 5. Modelos comparados (benchmarks)

Todos se entrenan **sin SMOTE ni remuestreo**, porque el remuestreo distorsiona las probabilidades.
Todos se **calibran con Platt** (`CalibratedClassifierCV(method='sigmoid', cv=3)`) dentro del pliegue
de ajuste. Así, la ventaja del modelo original no puede venir de una mala calibración de los
benchmarks, como el ECE de 0,11 de los finalistas con SMOTE del entregable 3.

1. **Los 7 modelos base del entregable 3:** kNN, Naive Bayes, regresión logística, árbol de decisión,
   Random Forest, XGBoost y SVM (RBF).
2. **Logística con splines por cuantiles.** El número de nudos y C se optimizan. Se incluye porque el
   piloto 2 mostró que es un benchmark fuerte.
3. **HistGradientBoosting** (scikit-learn).
4. **Métodos recientes del estado del arte:** LightGBM (Ke et al., 2017) y CatBoost
   (Prokhorenkova et al., 2018). Sus hiperparámetros iniciales son los recomendados en la literatura.
   Versiones fijadas: LightGBM 4.7.0 y CatBoost 1.2.10. El resto del entorno: Python 3.12.14,
   numpy 2.0.2, scipy 1.15.3, scikit-learn 1.6.1, xgboost 3.0.5 y optuna 4.2.1.
5. **Combinadores estándar** sobre los mismos expertos: promedio simple, pesos globales por EM
   ($K=1$) y stacking logístico. Se reportan también como parte de la ablación.

## 6. Modelo original y ablación

- **Expertos de RegionalMAPEM:** los benchmarks 1–4 (M = 11), con los hiperparámetros elegidos en el
  bucle interno de ese mismo pliegue externo.
- **Ajuste de los pesos sin fuga.** Los pesos $\pi$ se ajustan con predicciones fuera de pliegue de los
  expertos, obtenidas con cross-fitting de 3 pliegues dentro del pliegue de ajuste externo. Para
  predecir el pliegue externo, los expertos se reentrenan con todo el pliegue de ajuste.
- **Espacio de las regiones (regla general, la misma para los 4 conjuntos).** Se toman las variables
  numéricas continuas, es decir, las que tienen más de 2 valores distintos en el pliegue de ajuste. Se
  imputan por mediana, cada una se lleva a cuantiles normales (`QuantileTransformer`, salida normal) y
  se proyectan sobre las 2 primeras componentes principales. Todo se ajusta dentro del pliegue. La
  decisión se tomó sin mirar ninguna métrica de desempeño, por estas razones:
  1. **Transferibilidad.** La regla no depende de conocer el dominio, así que se aplica igual a los
     conjuntos adicionales. La guía (Cuadro 1) pide un mecanismo general.
  2. **Coherencia con los pilotos.** En el conjunto de desarrollo de Telco, las 2 componentes explican
     el 98 % de la varianza de las 4 variables continuas estandarizadas. La primera se correlaciona
     0,89 con `MonthlyCharges` y la segunda 0,74 con `tenure`, de modo que la regla es casi una rotación
     del plano (`tenure`, `MonthlyCharges`) de los pilotos 3–6. Esta revisión usó solo las variables,
     no la etiqueta.
  3. **Supuestos del GMM.** Las binarias (como `SeniorCitizen`) se excluyen porque producen
     covarianzas degeneradas en una mezcla gaussiana. La transformación a cuantiles normales evita que
     las variables asimétricas, como `TotalCharges` o los conteos de KDDCup09, obliguen a usar muchas
     componentes solo para modelar sus colas.
- **Hiperparámetros propios** (espacio exacto en `experiments/espacios.json`):
  - $K\in\{1,\dots,8\}$;
  - $c\in[0{,}01,\,200]$ en escala logarítmica ($c=0{,}01$ equivale en la práctica a no usar prior;
    la variante exacta $c=0$ va en la ablación);
  - covarianza del GMM: diagonal o completa;
  - criterio de parada: tolerancia relativa de $10^{-9}$ o 5000 iteraciones. Se subió de 500 a 5000
    antes del tag porque, en el piloto 6b, con 9 expertos el EM no alcanzaba la tolerancia en 500
    iteraciones (el ascenso sí fue monótono en todos los pliegues).
- **Ablación.** Cada variante se compara con RegionalMAPEM usando las mismas pruebas, y todas cuentan
  en la corrección de Holm:
  1. sin prior ($c=0$);
  2. sin regiones ($K=1$, pesos globales por EM);
  3. regiones duras (asignación por argmax);
  4. promedio simple;
  5. stacking logístico.

## 7. Presupuesto de optimización (equidad)

- **Optimizador:** Bayesiano TPE (Optuna), **30 evaluaciones por modelo y por pliegue externo**, de las
  cuales 8 son de arranque aleatorio.
- **Mismo objetivo para todos:** se minimiza el log-loss medio del bucle interno.
- **El modelo original** recibe las mismas 30 evaluaciones para sus propios hiperparámetros (K, c y la
  covarianza). Sus expertos son los benchmarks ya optimizados con su propio presupuesto, el mismo que
  reciben los combinadores estándar de la sección 5.5. Esta dependencia se reporta: el costo total del
  modelo original incluye el de sus expertos.
- **Espacios de búsqueda:** están en `experiments/espacios.json`. Los rangos del entregable 3 se pasan
  a dominios continuos y se amplían para la logística con splines, HistGB, LightGBM y CatBoost.
- **Punto de partida:** el primer ensayo de cada modelo es su configuración recomendada (valores por
  defecto de la librería o de la literatura citada en el JSON). Cuenta dentro de las 30 evaluaciones.
- **Reporte de cómputo:** número real de evaluaciones, tiempo de entrenamiento e inferencia, memoria y
  hardware de cada modelo.

## 8. Pruebas de hipótesis

Todas las pruebas son bilaterales, con α = 0,05.

| # | Prueba | Unidad | Comparación |
|---|---|---|---|
| T1 | t corregida de Nadeau-Bengio, con $n_\text{test}/n_\text{train}=1/4$ | 25 diferencias por pliegue externo | Original frente a cada benchmark |
| T2 | Permutación pareada (signos), 100 000 permutaciones | Pérdida por cliente, promediada sobre las 5 repeticiones (n = 5634) | Original frente a cada benchmark |
| T3 | Friedman con Iman-Davenport, Nemenyi post hoc y diagrama de diferencia crítica | Rangos en los 25 pliegues | Todos los modelos, original incluido |
| T4 | Wilcoxon de rangos con signo | 25 diferencias por pliegue | Original frente a cada benchmark |

- **Por qué estas pruebas.** DeLong no aplica al log-loss. McNemar se reporta solo como prueba
  secundaria sobre F1 en el hold-out.
- **Supuestos que se verifican:**
  - normalidad de las diferencias para T1 (Shapiro-Wilk y gráfico Q-Q);
  - simetría de las diferencias para T4;
  - intercambiabilidad bajo H0 para T2.
- **Tamaño del efecto:** δ de Cliff sobre los 25 log-loss por pliegue, con los umbrales de Romano et al.
  (efecto no despreciable si $|\delta|\ge 0{,}147$).
- **Intervalos:** bootstrap BCa (10 000 réplicas) de la diferencia media de log-loss.
- **Corrección por comparaciones múltiples:** Holm-Bonferroni sobre **todas** las comparaciones, que
  forman una sola familia:
  - T1, T2 y T4 del original frente a cada benchmark en log-loss;
  - los pares original-benchmark de Nemenyi (T3);
  - las mismas pruebas para F1 y AUC;
  - las 5 variantes de ablación;
  - las comparaciones en los conjuntos adicionales.

  La lista exhaustiva está en `experiments/familia_holm.json`, generada por
  `experiments/generar_familia_holm.py`. Son **223 comparaciones**:

  | Bloque | Comparaciones |
  |---|---|
  | Principal, log-loss, frente a los benchmarks (T1, T2 y T4) | 33 |
  | Principal, log-loss, frente a la ablación | 15 |
  | Principal, Friedman y pares de Nemenyi | 17 |
  | Principal, F1 y AUC | 44 |
  | Hold-out | 3 |
  | Conjuntos adicionales | 99 |
  | Entre conjuntos | 12 |

  - **Umbral más exigente:** con 223 comparaciones, el p más pequeño debe ser menor que
    0,05/223 ≈ 2,2 × 10⁻⁴. Por eso T2 usa 100 000 permutaciones.
  - **Exclusividad:** no se calcula ninguna prueba confirmatoria fuera de esa lista.
  - **Reporte:** siempre se reportan el p-valor crudo y el ajustado.
- **Entre conjuntos de datos:** Friedman con Iman-Davenport y Wilcoxon-Holm por pares sobre los 4
  conjuntos, además de la prueba bayesiana de rangos con signo. Las métricas de escalas distintas no
  se promedian. Se reconoce que con 4 conjuntos la potencia es baja.

## 9. Definición operativa de superioridad (R3)

Siguiendo la sección 6.4 de la guía, R3 se declara cumplido solo si se cumplen las cuatro condiciones
a la vez:

1. El log-loss medio externo de RegionalMAPEM es menor que el del mejor benchmark y que el del mejor
   de cada uno de los 7 modelos base.
   Las combinaciones del dashboard (seleccionadas por F1 en el entregable 3, varias con SMOTE) también se
   evalúan en log-loss sobre los mismos pliegues y se reportan. La comparación decisiva, sin embargo, es
   frente a las versiones reajustadas y calibradas de este protocolo, porque compararse con modelos mal
   calibrados daría una ventaja injusta.
2. Al menos tres pruebas distintas entre T1, T2, T3 y T4 rechazan H0 frente al mejor benchmark, con
   p ajustado por Holm < 0,05.
3. $|\delta_\text{Cliff}|\ge 0{,}147$ frente al mejor benchmark.
4. Se cumple H2 en los conjuntos adicionales.

Si alguna condición falla, se reporta **R3 no alcanzado** y se analiza la causa probable. El análisis
incluye la cota del oráculo, la redundancia entre expertos y el techo de información del dataset,
contrastado con la auditoría publicada en arXiv (2026).

## 10. Robustez y calidad probabilística

Se aplican a todos los modelos con el mismo protocolo y tienen carácter exploratorio:

- **Sensibilidad:** curvas del log-loss al variar K y c.
- **Estabilidad:** 10 semillas de modelo, `seed_for('model', s)` con $s=0,\dots,9$, sobre la
  repetición 0.
- **Ruido:** se contamina el 5 %, 10 % y 20 % de las variables numéricas con ruido gaussiano
  ($0{,}5\,\sigma$) y se invierte el 5 % y el 10 % de las etiquetas, solo en el pliegue de ajuste.
- **Tamaño de muestra:** curva de aprendizaje con el 10 %, 25 %, 50 % y 100 % del pliegue de ajuste, y
  tiempos al variar n y p.
- **Cambio de distribución:** evaluación única sobre el hold-out y una partición por `tenure`
  (entrenar con tenure ≤ 24 y evaluar con tenure > 24).
- **Calibración:** curva de calibración, ECE, Brier y log-loss.
- **Incertidumbre:** predicción conforme partida, condicionada por clase (Mondrian), con cobertura
  nominal del 90 % y del 95 %. Se reportan la cobertura empírica y el tamaño medio de los conjuntos.
- **Análisis de errores:** desempeño por región, tipo de contrato y rango de `tenure`, frente al mejor
  benchmark.

## 11. Registro de pilotos e iteraciones de diseño (antes del tag)

| # | Fecha | Quién | Qué se probó | Resultado | Decisión |
|---|---|---|---|---|---|
| 1 | ≤ 2026-11-03 | Laura | SVM+ (LUPI) con CLTV y Churn Score como información privilegiada | F1 de 0,35 frente a 0,58 sin información privilegiada; riesgo de fuga, y las variables no pertenecen al dataset de Kaggle | Descartado (de acuerdo con el docente) |
| 2 | ≤ 2026-11-03 | Laura | Modelos únicos sin antecedentes (NSC, credal, Mondrian) | Novedad solo de aplicación | Descartados |
| 3–5 | 2026-11-03 | Laura | 7 variantes de la ponderación por regiones con MAP-EM (3 iteraciones; se agregaron CatBoost y LightGBM) | Corridas por error con la versión extendida de IBM y los 7043 clientes (hold-out incluido); cifras descartadas | Se elige la familia; exposición del hold-out declarada en la sección 4.1 |
| 6 | 2026-11-07 | Nataly | Cota del oráculo (`pilotos/piloto_oraculo.py`) | Ganancia honesta ≈ 0,0007; las regiones no mejoran a los pesos globales fuera de la muestra; ascenso de MAP-EM monótono en todos los pliegues | Se mantiene el modelo y se motiva el prior como protección |
| 6b | 2026-11-07 | Laura | Repetición con el dataset del proyecto, solo desarrollo (5634), regla de regiones de la sección 6, 9 expertos y ablación (`experiments/pilotos/piloto_kaggle/`) | Log-loss 0,4116 frente a 0,4128 de la logística con splines; solo Wilcoxon significativo frente a ella; las regiones aportan 0,0001 frente a K = 1; ascenso monótono, pero sin alcanzar la tolerancia en 500 iteraciones | Se mantiene el modelo; `max_iter` sube a 5000; se espera R3 no alcanzado |
| 7 | 2026-11-07 | Nataly | Híbrido GLM con splines + boosting residual (`pilotos/piloto_hibrido.py`) | Ganancia de 0,0020, de la cual ≈ 0,0014 la logra una logística con más nudos | Descartado; la logística con splines entra como benchmark |

## 12. Desviaciones

Toda desviación posterior al tag se registra aquí, con fecha, motivo y efecto esperado, y se declara en
el capítulo 6 del libro. Los análisis no previstos se reportan como exploratorios.

## 13. Congelamiento

- Pendiente antes del tag:
  - obtener la revisión de ambas autoras;
  - medir el tiempo de un pliegue externo completo (todos los modelos, 30 evaluaciones) y, si el diseño
    no cabe en el plazo, reducirlo aquí (por ejemplo, menos repeticiones en los conjuntos adicionales);
  - confirmar si las menciones a "entregable 3" en las secciones 4.1 y 5 deben decir "entregable 2";
  - repetir la búsqueda por citación en WoS y Google Académico (De Caigny 2018, Soares 2006, Yan 2001).
- Ya resuelto:
  - Iranian Churn (UCI 563) verificado por Nataly;
  - regla del espacio de regiones fijada (sección 6);
  - `espacios.json` y `familia_holm.json` escritos;
  - LightGBM y CatBoost instalados.
- Tag: `protocolo-v1` · Fecha: ____ · Commit: ____
