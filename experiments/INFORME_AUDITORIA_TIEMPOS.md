# Informe técnico para auditar tiempos de ejecución

Proyecto Telco Churn, Entregable 4. Corte de la inspección: 2026-11-08, 21:01 a 21:09 (hora de Colombia, UTC-5).

**Alcance de la inspección.** Solo lectura. No se modificó código, no se detuvo el entrenamiento y no se corrió ningún experimento. Se leyeron checkpoints y se calculó el tamaño de las matrices tras el preprocesamiento, con un proceso breve y sin entrenar modelos. Los únicos archivos nuevos son este informe y tres archivos auxiliares en el directorio temporal de la sesión (sección 10).

**Cómo leer las etiquetas**

| Etiqueta | Significado |
|---|---|
| **[V]** | Verificado hoy en la máquina, en los archivos o en los registros |
| **[D]** | Derivado por cálculo a partir del código o de datos verificados |
| **[E]** | Estimación, con supuestos que pueden fallar |
| **[S]** | Viene del historial de la sesión; no se volvió a comprobar hoy |
| **[NC]** | No se pudo comprobar |

**Regla para quien audite.** El experimento está preregistrado. No use métricas de desempeño para elegir optimizaciones, porque eso contaminaría el análisis. Este informe no reporta ninguna. `log_anidado.txt` contiene valores del bucle interno en sus primeras 4143 líneas, de antes de silenciar Optuna, y algunos aparecieron en la inspección de hoy. Ninguna decisión de este informe depende de ellos.

## Resumen para auditar

1. **Avance.** Hay 465 de 1020 tareas guardadas, pero eso equivale a solo un 12 % a 19 % del cómputo **[E]**. Lo más caro (KDDCup09) va al final.
2. **Fin del cómputo.** Con 10 procesos, la validación anidada terminaría entre la noche del 9 y el mediodía del 10 de noviembre **[E]**. Todo el cómputo terminaría entre el 10 de noviembre por la mañana y el 11 temprano. Es una estimación. El plazo real es la entrega del **2026-11-15**.
3. **Hallazgo principal, sin confirmar.** Los 8 hilos eficientes muestran unos **360 MHz efectivos** con carga de 70 % a 84 %, frente a unos 2050 MHz de los hilos de rendimiento **[V como lectura del contador; causa NC]**. El plan de energía es "Equilibrado". Si la lectura es real, esta es la mayor oportunidad, y se prueba sin tocar código ni resultados.
4. **Riesgo principal.** KDDCup09 aún no se ha ejecutado en la corrida y hay solo 7,5 GB de RAM. Con 16 procesos ya hubo una caída por falta de recursos.
5. **Costo no previsto.** SHAP costaría unas 6 horas en un solo proceso **[E]** y además no fija semilla **[V]**, así que hoy no sería reproducible.

## 1. Tareas y requisitos pedidos

Las fuentes primarias son la guía del Entregable 4, el correo del profesor y el libro del curso. Aquí se resumen desde el historial **[S]**. No se releyeron hoy, así que no las sustituye este resumen.

**Proyecto.** Predicción de churn en Telco, de Nataly Cárdenas y Laura Rivera, para la Universidad del Norte (profesor Lihki), documentada en un Jupyter Book. Entrega: 2026-11-15.

**Restricción rectora.** No hacer nada incorrecto en machine learning. Si algo pone en riesgo la validez, se prefiere aceptar una nota menor a 5.

**Criterios del profesor**

- R1: la novedad está en la regla, con matriz de novedad que cubra mezcla de expertos con compuerta EM, selección dinámica de ensambles, stacking con pesos locales y ensambles ponderados.
- Prueba de convergencia con supuestos.
- Log-loss permitido si se preregistra y se declara que se eligió después de los pilotos.
- Mismo presupuesto para todos, con calibración Platt o isotónica en el bucle interno.
- F1 y AUC reportados y contados en Holm.
- Wilcoxon o Nadeau-Bengio, permutación pareada, Friedman-Nemenyi y δ de Cliff ≥ 0,147.
- Un resultado negativo honesto tiene tope de 4,0.
- SVM+ con Churn Score queda descartado.

**Último pedido de la usuaria antes de este informe**

1. Resolver los pendientes posibles sin errores y listar los que no pueda resolver o de los que no esté seguro.
2. Ejecutar el protocolo.
3. Seguir siempre la guía del Entregable 4 y tener en cuenta el libro del profesor.
4. Escribir el libro con tono humano, directo y conciso, para lectores generales y profesionales, y explicar el porqué de las decisiones.
5. No usar frases cliché, emojis ni referencias a IA, instrucciones, guías o recursos. La excepción es lo que la guía exija declarar, como el uso de IA en el capítulo final.
6. Mostrar todas las salidas con el estilo de tablas del libro, nunca con el formato por defecto.
7. Al tener el resultado "final", auditar de forma exhaustiva contra el protocolo y la guía, y agregar lo que falte salvo que dañe la veracidad, para llegar a la máxima nota posible.

**Pedido de este informe.** Los 10 puntos que siguen, sin modificar código, sin detener entrenamientos y sin iniciar experimentos.

## 2. Estado de las tareas

### Completadas

| Tarea | Evidencia |
|---|---|
| Protocolo, espacios de búsqueda y familia de Holm (223 comparaciones) | Archivos presentes **[V]** |
| Commit `77fa533bf` y tag local `protocolo-v1`, 2026-11-08 16:13:11 | `git log`, `git tag` **[V]** |
| Clase `RegionalMAPEM`, con `map_em` acelerado 4,6 veces e idéntico hasta 2e-15 | **[S]** |
| 20 pruebas unitarias aprobadas antes del tag | **[S]**; no se repitieron hoy |
| Medición previa de tiempos | `tiempos_medidos.csv` **[V]** |
| Búsqueda de literatura en OpenAlex y arXiv | `experiments/literatura/` **[V]** |
| Validación anidada de Telco: 275 tareas de benchmark y 25 de combinadores | 300 pkl **[V]** |

### En ejecución

| Dato | Valor |
|---|---|
| Tarea en segundo plano | `buln1ow5x`, con `correr_con_reintentos.sh 10` **[V]** |
| Inicio del intento vigente | 2026-11-08 17:19:41 **[V]** |
| Conjunto actual | `uci_563_iranian` **[V]** |
| Tareas guardadas a las 21:08:45 | 465 de 1020 **[V para 465; D para 1020]** |
| Trabajadores | 10, más el proceso principal y un auxiliar (12 python) **[V]** |
| Reinicios desde 17:19:41 | Ninguno: solo aparece "intento 1" en el log **[V]** |
| Evaluaciones fallidas | 0 de 13 110 en benchmarks, y 0 de 750 en RegionalMAPEM de Telco **[V]** |

### Pendientes de cómputo, en orden

| # | Tarea | Comando | Estado del script |
|---|---|---|---|
| 1 | Terminar Iranian, OpenML 40701 y KDDCup09 | ya en curso | En ejecución |
| 2 | Combinaciones del tablero | `complementos_confirmatorio.py dashboard` | **Sin probar** |
| 3 | Robustez y sensibilidad | `robustez.py N` | Sin ejecutar |
| 4 | Corrida de tenure | `correr_finales.py tenure N` | Sin ejecutar |
| 5 | Compilación parcial | `analizar.py` | **Sin probar** |
| 6 | Hold-out, una sola vez | `correr_finales.py holdout N` | Sin ejecutar |
| 7 | Compilación final | `analizar.py` | **Sin probar** |
| 8 | Escalabilidad y SHAP | `complementos_confirmatorio.py escalabilidad` y `shap` | **Sin probar** |

### Pendientes sin cómputo

- Registrar en la sección 12 del protocolo, hoy vacía **[V]**: la caída, el paso de 16 a 10 procesos, el silencio de los registros de Optuna y los scripts escritos después del tag.
- Hacer commit de lo posterior al tag: `correr_anidado.py` modificado, y `analizar.py`, `complementos_confirmatorio.py` y `correr_con_reintentos.sh` sin versionar **[V]**.
- Escribir resultados en el libro y actualizar `MODELO_PROPIO.ipynb`.
- Completar capítulos: trazabilidad, literatura, validación estadística, evaluación y robustez, tablero, conclusiones y reproducibilidad **[S]**.
- Crear archivos faltantes: `MODEL_CARD.md`, `DATA_STATEMENT.md`, `CITATION.cff`, `LICENSE`, `references.bib`, `artifacts/` y la tabla maestra de experimentos **[S]**.
- Revisar y desplegar el tablero, y hacer la auditoría final.

### Requieren a las autoras

- **Publicar el preregistro.** El tag y el commit `77fa533bf` **no están en GitHub**: el remoto sigue en `e1ecf2be5` y no tiene el tag **[V]**. Hasta subirlos no hay evidencia pública de la fecha de congelamiento.
- Revisión de Laura de los cambios del 2026-11-08.
- Búsquedas en WoS y Google Académico, y lectura a texto completo de los artículos cercanos de 2010, 2012, 2024 y 2025.
- Comentarios literales del profesor para la tabla de trazabilidad.
- Cuentas de Render y Zenodo, elección de licencia y firma de la declaración.

## 3. Por qué "10 de noviembre" y qué tan firme es

**Es una estimación, no un plazo.** El único plazo real es la entrega del 2026-11-15.

**De dónde salió.** Coincide con este cálculo: 430 horas-proceso medidas ÷ 10 procesos = 43 h, contadas desde las 17:19 del 8 de noviembre, lo que da el 10 de noviembre hacia las 12:00 **[D]**. El cálculo no quedó guardado en un archivo, así que esto es una reconstrucción. Se hizo antes de tener tiempos reales.

**Qué cambia con los datos reales.** La medición previa sobrestimó Telco: se midieron 61 h-proceso y se gastaron 29,8 en benchmarks, un factor de 0,49 **[V]**. Pero el factor varía mucho entre modelos, de 0,37 en CatBoost a 1,72 en KNN, y no se puede asumir igual en KDDCup09.

La tabla usa 9 procesos efectivos, que es el ritmo observado después del reinicio (sección 7).

| Etapa | Horas-proceso | Reloj | Fin estimado | Tipo |
|---|---|---|---|---|
| Telco, incluida la caída | 29,8 + combinadores | 3,5 h | 8 nov, 19:43 | **[V]** |
| Iranian, lo que falta | 8 a 16 | 1,5 a 2,5 h con combinadores | 8 nov, 22:30 a 23:30 | **[E]** |
| OpenML 40701 | 46 a 98 | 5,6 a 11,4 h | 9 nov, 04:00 a 11:00 | **[E]** |
| KDDCup09 con 10 procesos | 141 a 232 | 16 a 26 h | 9 nov, 20:00 a 10 nov, 13:00 | **[E]** |
| KDDCup09 si la RAM obliga a 5 procesos | 141 a 232 | 31 a 52 h | 10 nov, 11:00 a 11 nov, 15:00 | **[E]** |
| Robustez y sensibilidad | 8,7 | 1 h | después | **[E]** |
| Tenure y hold-out | 2,8 a 4 | 2 a 3 h | después | **[E]** |
| SHAP | 6 o más | 6 h en un solo proceso | después | **[E]** |
| Tablero, escalabilidad, análisis | sin medir | menos de 2 h, sin base firme | después | **[NC]** |

**Fin total del cómputo [E]**

| Escenario | Fin aproximado |
|---|---|
| Favorable, con 10 procesos | 10 nov, por la mañana |
| Desfavorable, con 10 procesos | 11 nov, temprano |
| KDDCup09 limitado a 5 procesos | 12 nov, temprano |

**Qué puede invalidar estas cifras**

- La proyección de KDDCup09 parte de 6 evaluaciones al azar por modelo. TPE no muestrea igual.
- No hay datos de uso de memoria de KDDCup09 durante la corrida.
- Una caída pierde las tareas en curso.
- Los tiempos de combinadores fuera de Telco no se han medido.
- La máquina debe seguir encendida, enchufada y sin suspenderse.

## 4. Modelos, configuraciones, pliegues y entrenamientos

**Modelos [V].** 11 benchmarks: `knn`, `naive_bayes`, `logistic`, `decision_tree`, `random_forest`, `xgboost`, `svm`, `logistic_splines`, `hist_gb`, `lightgbm`, `catboost`. Además, `RegionalMAPEM` y sus comparaciones: `mapem_sin_prior`, `mapem_K1`, `mapem_regiones_duras`, `promedio_simple`, `stacking_logistico`.

**Búsqueda [V].** TPE de Optuna con 30 evaluaciones por modelo y pliegue externo, 8 de arranque aleatorio y la primera fijada en la configuración recomendada. Sin poda. Como la búsqueda es adaptativa y sobre rangos continuos, no hay una rejilla fija de configuraciones.

**Pliegues [V].**

| Conjunto | Repeticiones × externos | Pliegues externos | Internos |
|---|---|---|---|
| Telco | 5 × 5 | 25 | 3 |
| Iranian | 5 × 5 | 25 | 3 |
| OpenML 40701 | 5 × 5 | 25 | 3 |
| KDDCup09 | 2 × 5 | 10 | 3 |
| **Total** | | **85** | |

**Ajustes por tarea de benchmark [D].** Cada pipeline calibrado con `cv=3` ajusta 3 veces, porque `CalibratedClassifierCV` en sklearn 1.6.1 no hace un ajuste final adicional con su valor por defecto:

30 evaluaciones × 3 internos × 3 de calibración + 3 del reajuste final = **273 ajustes**

**Volumen total [D]**

| Bloque | Tareas | Evaluaciones de hiperparámetros | Ajustes |
|---|---|---|---|
| Benchmarks anidados: 85 × 11 | 935 | 28 050 | 255 255 |
| Combinadores: 85 | 85 | 5100 | 9010 RegionalMAPEM y 7990 logísticas |
| Tenure y hold-out | 24 | 780 | 6006 de benchmarks, más unos 400 de combinadores |
| Robustez: 21 escenarios × 5 × 11 | 1155 + 105 | 0 | 13 860 de benchmarks, más 315 de combinadores |
| Sensibilidad: 25 × 8 × 8 | 25 | 0 | 1600 RegionalMAPEM |
| Tablero: 7 × 25 | 175 | 0 | unos 225 |
| Escalabilidad y memoria | 107 | 0 | unos 300 |
| SHAP | 1 | 0 | 34, más unos 6 millones de filas predichas |
| **Total** | | **33 930** | **unos 296 000** |

Cada ajuste de RegionalMAPEM incluye además una mezcla gaussiana con `n_init=3` y hasta 5000 iteraciones de EM **[V]**. Se alcanzó ese máximo en Telco **[V]**.

## 5. Datos y pipeline

Dimensiones medidas hoy sobre la repetición 0, pliegue 0 **[V]**. En KDDCup09, las columnas finales cambian según el pliegue.

| Conjunto | Filas | Variables crudas | Numéricas / categóricas | % churn | % celdas faltantes | Ajuste / evaluación | Columnas finales | Matriz densa |
|---|---|---|---|---|---|---|---|---|
| Telco (desarrollo) | 5634 | 19 | 3 / 16 | 26,5 | 0,0 | 4507 / 1127 | 24 | 0,9 MB |
| Iranian | 2850 | 13 | 13 / 0 | 15,7 | 0,0 | 2280 / 570 | 13 | 0,2 MB |
| OpenML 40701 | 5000 | 19 | 17 / 2 | 14,1 | 0,0 | 4000 / 1000 | 70 | 2,2 MB |
| KDDCup09 (submuestra) | 10 000 | 230 | 192 / 38 | 7,3 | 69,8 | 8000 / 2000 | 902 | 57,7 MB |

Además hay un hold-out de Telco de 1409 clientes, que no se ha evaluado.

**Pipeline [V]**

1. Telco: ETL por filas (`HistoricalFeatures`) y luego `StandardScaler` más one-hot con `drop="first"`.
2. Otros conjuntos: imputación por mediana o moda, `StandardScaler` y one-hot con `max_categories=50`. En KDDCup09, primero se eliminan columnas con más de 95 % de faltantes.
3. Salida **densa** en todos los casos (`sparse_output=False`).
4. Modelo con hiperparámetros de `espacios.json`.
5. Calibración Platt con `cv=3`. El preprocesamiento va **dentro** de la calibración, de modo que se reajusta en cada partición. Esto lo exige el protocolo para evitar fuga.
6. Sin SMOTE ni pesos de clase.

**RegionalMAPEM [V].** Se ajusta sobre las predicciones fuera de pliegue de los 11 expertos, sin reentrenarlos. Sus regiones usan PCA de 2 componentes y una mezcla gaussiana propia. Busca K de 1 a 8, c de 0,01 a 200 en escala logarítmica, y covarianza `diag` o `full`.

## 6. Hardware y entorno

| Componente | Valor | Tipo |
|---|---|---|
| CPU | Intel Core Ultra 5 125H, 14 núcleos, 18 hilos | **[V]** |
| Reparto de núcleos | 4 de rendimiento (8 hilos), 8 eficientes y 2 de baja potencia | Especificación pública, coherente con los 3 grupos de hilos observados **[D]** |
| RAM | 7,52 GB visibles, en **1 solo módulo** de 8 GB a 5600 MT/s | **[V]** |
| Canal de memoria | Un módulo implica canal único | **[D]** |
| Archivo de paginación | 23,5 GB, con pico de 5,9 GB | **[V]** |
| GPU | Solo Intel Graphics integrada. No hay GPU dedicada detectada | **[V]** |
| Disco | SSD SK hynix de 512 GB, 222,7 GB libres en C: | **[V]** |
| Sistema | Windows 11 Home Single Language 10.0.26200 | **[V]** |
| Energía | Conectada a corriente, plan "Equilibrado" | **[V]** |
| Equipo | Local, con servicio de OMEN Command Center. Su perfil térmico activo: **[NC]** | **[V]** para el servicio |
| Python | 3.12.14, entorno conda `ml_project` | **[V]** |
| Librerías | numpy 2.0.2, scipy 1.15.3, sklearn 1.6.1, xgboost 3.0.5, optuna 4.2.1, lightgbm 4.7.0, catboost 1.2.10, joblib 1.5.3, shap 0.48.0 | **[V]** |
| Temperatura y límite de potencia | Sin lectura disponible | **[NC]** |

### Memoria durante la inspección [V]

| Indicador | Valor |
|---|---|
| RAM disponible | 460 a 840 MB |
| Memoria comprometida | 16 a 17 GB, de un límite de 32,8 GB |
| Privada de python (12 procesos) | 2,65 GB |
| Conjunto de trabajo de python | 0,5 a 0,66 GB |
| Privada de `claude` (12 procesos) | 2,3 GB |
| Privada de `ChatGPT` (15 procesos) | 1,7 GB |
| Otras apps: Chrome, Edge WebView, Defender | 0,45, 0,44 y 0,44 GB |
| Proceso de compresión de memoria | 1,3 GB |

Solo una quinta parte de la memoria privada de python está en RAM en cada momento, y hubo ráfagas de paginación: hasta 65 893 páginas de salida y 16 463 de entrada por segundo, con el disco ocupado en esa muestra. Hay presión de memoria real. Su costo en tiempo no se ha medido **[NC]**.

### Frecuencia durante la inspección [V como lectura]

12 muestras, una cada 5 s, entre 21:07 y 21:08:

| Grupo | Frecuencia efectiva | Carga |
|---|---|---|
| Hilos 0-1 y 10-15 (rendimiento) | 1884 a 2190 MHz | 63 a 100 % |
| Hilos 2-9 (eficientes) | 287 a 385 MHz | 70 a 100 % |
| Hilos 16-17 (baja potencia) | 151 a 301 MHz | 0 a 3 % |

Es frecuencia base × `% Processor Performance`. Este contador puede engañar en CPU híbridas, y una ventana de un minuto no basta para concluir **[NC]**.

## 7. Qué consume más tiempo: mediciones reales

Todos los tiempos son de reloj (`perf_counter`), por tarea, con competencia entre procesos. El código **no registra tiempo de CPU ni memoria por tarea** **[V]**.

### Telco, 25 pliegues completos [V]

| Modelo | s/tarea, media | s/tarea, máximo | Evaluación, mediana (s) | Evaluación, máximo (s) | h-proceso | % | Real / medido |
|---|---|---|---|---|---|---|---|
| catboost | 1803 | 3335 | 17,2 | 1203 | 12,52 | 42,0 | 0,37 |
| random_forest | 784 | 1334 | 20,0 | 131 | 5,44 | 18,3 | 0,49 |
| hist_gb | 570 | 1029 | 12,5 | 131 | 3,96 | 13,3 | 0,56 |
| svm | 278 | 445 | 7,2 | 28 | 1,93 | 6,5 | 1,33 |
| lightgbm | 249 | 571 | 6,5 | 77 | 1,73 | 5,8 | 0,49 |
| xgboost | 214 | 416 | 5,9 | 28 | 1,49 | 5,0 | 0,83 |
| knn | 127 | 254 | 3,7 | 16 | 0,88 | 3,0 | 1,72 |
| logistic | 102 | 213 | 2,1 | 17 | 0,71 | 2,4 | 1,36 |
| logistic_splines | 62 | 98 | 1,7 | 9 | 0,43 | 1,5 | 1,04 |
| decision_tree | 54 | 136 | 1,4 | 6 | 0,38 | 1,3 | 1,53 |
| naive_bayes | 51 | 105 | 1,4 | 5 | 0,35 | 1,2 | 1,40 |
| **Total** | | | | | **29,82** | | **0,49** |

Tres modelos concentran el 73 % del costo. CatBoost tiene una cola muy larga: su peor evaluación es 70 veces la mediana.

**Combinadores de Telco [V].** Búsqueda de RegionalMAPEM de 662 s en media y 900 s como máximo, con 21,5 s de mediana por evaluación. Las 25 tareas tomaron 35 minutos, de 19:08 a 19:43.

### Iranian, parcial [V]

Hasta el corte hay 162 tareas y 9,34 h-proceso, con CatBoost en 54 % y solo 7 de 25 pliegues terminados. Random forest lleva 18 % y hist_gb 13 %. La media de CatBoost es 2591 s y está sesgada, porque aún faltan 18 tareas.

### KDDCup09, proyección [E]

Sin datos reales todavía. Horas-proceso por modelo para 10 pliegues:

| Modelo | Sin factor de corrección | Con factor de Telco | Con factor de Iranian |
|---|---|---|---|
| random_forest | 93,7 | 46,2 | 46,2 |
| catboost | 49,8 | 18,2 | 34,1 |
| hist_gb | 36,9 | 20,5 | 20,6 |
| svm | 23,6 | 31,3 | 21,7 |
| xgboost | 9,4 | 7,8 | 6,6 |
| knn | 8,0 | 13,7 | 5,6 |
| Otros cinco | 10,2 | 10,2 | 6,8 |
| **Total** | **231,6** | **148,0** | **141,5** |

### Efecto de la cantidad de procesos [D]

Media por tarea de Telco, según el momento en que terminó:

| Modelo | 16 procesos (s) | 10 procesos (s) | Razón |
|---|---|---|---|
| catboost | 2463 | 1639 | 1,50 |
| random_forest | 1018 | 627 | 1,62 |
| hist_gb | 787 | 400 | 1,97 |
| svm | 364 | 211 | 1,72 |
| xgboost | 276 | 157 | 1,76 |
| naive_bayes | 65 | 35 | 1,88 |

Al expresar el trabajo terminado en segundos de 10 procesos:

- Con 16 procesos: unos 7,5 procesos efectivos, o hasta unos 11 si se cuenta el trabajo perdido en la caída **[E]**.
- Con 10 procesos: unos 9,1.

**No se observó una ganancia clara con 16 procesos, y se perdió estabilidad.** No fue un experimento controlado: cambian los pliegues, la mezcla de modelos y el momento del día. Tampoco separa el efecto de la RAM del de los núcleos híbridos.

### Registros disponibles [V]

| Archivo | Contenido |
|---|---|
| `experiments/resultados/log_anidado.txt` | Una línea con la hora por tarea. Registros de Optuna hasta la línea 4143. Traceback de `WinError 1450` en las líneas 4144 a 4156 |
| `experiments/resultados/crudos/<conjunto>/r<R>_f<F>/<modelo>.pkl` | `seg_busqueda`, `seg_ajuste`, `seg_pred`, `n_pred`, `ensayos`, `fallidos` y `registro`, con los segundos de cada evaluación |
| `experiments/confirmatorio/tiempos_medidos.csv` | Medición previa al tag |
| Archivos auxiliares de la sección 10 | Tiempos por tarea, ya tabulados |

**Caída registrada [V].** Poco después de las 17:17:46, con 16 procesos: `OSError: [WinError 1450] Recursos insuficientes`. Se conservaron los checkpoints. Lo que estaba en curso se perdió, sin que se pueda cuantificar con exactitud.

## 8. Paralelismo, búsqueda, caché y persistencia

| Aspecto | Configuración | Tipo |
|---|---|---|
| Nivel de paralelismo | Por tarea (conjunto, repetición, pliegue, modelo), con joblib y `loky` | **[V]** |
| Procesos | 10 por argumento. Valor por defecto del script: 16 | **[V]** |
| Lotes | `batch_size=1`, `pre_dispatch` por defecto (`2*n_jobs`) | **[V]** |
| Hilos por modelo | RF, XGBoost y LightGBM con `n_jobs=1`. CatBoost con `thread_count=1` | **[V]** |
| Variables de entorno | `OMP`, `OPENBLAS`, `MKL` y `NUMEXPR_NUM_THREADS` a 1 con `setdefault` en `comun.py` | **[V]** |
| Uso real por trabajador | 12 842 s de CPU en 13 300 s de reloj: 97 % de un núcleo | **[D]** |
| Optuna | Secuencial dentro de cada tarea, en memoria, sin `storage`, sin poda | **[V]** |
| Semillas | `SeedSequence([42, rol, ...])` por tarea, independientes del orden | **[V]** |
| Caché de preprocesamiento | **Ninguna**: se reajusta en cada partición, como pide el protocolo | **[V]** |
| Caché de predicciones | Las de los expertos se guardan y las reutilizan los combinadores | **[V]** |
| Persistencia | Un pkl por tarea, con escritura atómica (`.tmp` y `replace`) | **[V]** |
| Reanudación | Se salta lo que ya existe. No hay checkpoint por evaluación, así que una caída pierde la tarea en curso | **[V]** |
| Reintentos | Hasta 60, con 60 s de espera | **[V]** |
| Peso de checkpoints | 22,7 MB en 460 archivos: despreciable | **[V]** |
| Reparto entre etapas | Cada conjunto corre benchmarks, luego combinadores, y solo entonces pasa al siguiente | **[V]** |
| SVM | `cache_size=500` MB por ajuste | **[V]** |
| Determinismo real | Previsto por diseño. **No se ha comprobado repitiendo una tarea** | **[NC]** |

Los trabajadores se recrearon a las 19:08 y a las 19:43, cuando cambió la etapa **[V para hora y coincidencia; causa NC]**.

## 9. Cuellos de botella y optimizaciones

### Cuellos de botella

| # | Cuello de botella | Evidencia | Peso |
|---|---|---|---|
| 1 | Posible limitación de hilos eficientes a unos 360 MHz | Sección 6 | Potencialmente el mayor; **sin confirmar** |
| 2 | RAM escasa, canal único y otras apps abiertas | Sección 6 y la caída | Alto, y limita los procesos |
| 3 | Modelos de árboles con muchas iteraciones | Sección 7 | Alto; es inherente al diseño |
| 4 | KDDCup09 densa, con 902 columnas | Sección 5 | Cerca de dos tercios del cómputo restante **[E]** |
| 5 | SHAP con KernelExplainer en serie | 200 × 300 × 100 = 6 millones de filas, a 3,58 s por 1000 filas sumando los 11 modelos | Unas 6 h **[E]** |
| 6 | Reajuste repetido del preprocesamiento | Cota abajo | Bajo |

### Opciones ordenadas por relación entre ganancia y riesgo

Ninguna reduce el alcance ni cambia la evaluación. **Ninguna se ha aplicado.**

| # | Opción | ¿Toca código? | ¿Cambia resultados? | Ganancia | Riesgo y condición |
|---|---|---|---|---|---|
| 1 | Probar el plan "Máximo rendimiento" o el perfil de rendimiento de OMEN, y volver a medir frecuencia y ritmo | No | No: la frecuencia no altera los cálculos | Posiblemente grande **[NC]** | Temperatura; debe hacerlo la usuaria. Los tiempos antes y después se reportan separados |
| 2 | Cerrar ChatGPT, Chrome y otras apps prescindibles | No | No | Libera hasta unos 2,5 GB de privada **[V para el tamaño]**; efecto en tiempo **[NC]** | Ninguno. No cerrar esta sesión |
| 3 | Mantener 10 procesos y vigilar RAM al iniciar KDDCup09 | No | No | Evita caídas | No subir sin medir |
| 4 | Corregir `n_jobs=16` fijo en `complementos_confirmatorio.py` y pasar `N` explícito a `robustez.py` | Sí | No | Evita otra caída | Está **pendiente antes de correr esos pasos**. Registrar como desviación |
| 5 | SHAP con semilla fija y un trabajo por cliente | Sí | Es exploratorio; hoy tampoco es reproducible | De unas 6 h a 1 o 2 h **[E]** | Cada proceso carga 11 modelos: vigilar RAM |
| 6 | Controlar la concurrencia en KDDCup09: 10 procesos para modelos ligeros y menos para RF, CatBoost y SVM | Script aparte | No, por semillas por tarea | Evita caídas y paginación | Comprobar igualdad con una tarea ya guardada |
| 7 | Correr KDDCup09 por pliegues en un segundo equipo, como el de Laura | Mínimo | Puede variar por CPU | Muy alta | Mismas versiones; todos los modelos de un pliegue en la misma máquina para mantener el pareado; prueba de igualdad previa; desviación declarada |
| 8 | Añadir memoria, si el equipo lo permite | No | No | Alta | Ampliable o no: **[NC]** |
| 9 | Caché del preprocesamiento por partición | Sí | No, si se implementa bien | Cota: hasta 7 % a 11 % en KDDCup09 **[D]** y menos en los demás | **No se recomienda a mitad de corrida**: toca lo que protege contra la fuga |
| 10 | Excluir carpetas de Defender | No | No | Probablemente baja **[NC]** | Cambia seguridad. Decisión de la usuaria; no se recomienda sin medir |

La cota de la opción 9 sale de usar Naive Bayes como techo: 5,65 s por evaluación interna × 90 × 110 tareas = 15,5 h-proceso, frente a 141 a 232.

### Cómo comprobar que un cambio no altera resultados

Repetir una tarea barata ya guardada, con `RESULTADOS_DIR` apuntando a otra carpeta, y exigir igualdad exacta en `oof`, `pte` y `params`. Empezar por `decision_tree` en Telco r0 f0. Debe hacerse con la corrida principal en marcha solo si hay RAM suficiente. Esto es también la prueba de determinismo pendiente, y **no se ejecutó** porque se pidió no iniciar nada.

### Lo que no debe hacerse

Cada una de estas opciones cambia lo preregistrado o rompe la comparación entre modelos, porque ya hay pliegues terminados con la configuración actual:

- Bajar evaluaciones, pliegues, repeticiones o iteraciones.
- Usar poda, Successive Halving o parada temprana.
- Reducir los espacios de búsqueda o quitar modelos.
- Sacar el preprocesamiento de la calibración.
- Cambiar a matrices dispersas, GPU, `sklearnex` o más hilos en XGBoost, LightGBM, CatBoost o HistGB, salvo que se pruebe igualdad exacta. Estos cambios pueden alterar los resultados numéricos.
- Evaluar el hold-out más de una vez.

### Problemas detectados que no son de tiempo

- **SHAP sin semilla.** `shap/explainers/_kernel.py` usa `np.random` global y `correr_shap()` no fija semilla **[V]**.
- **`analizar.py`** tiene una expresión `if False else` en la línea de McNemar para F1 que hay que limpiar **[S]**.
- **Protocolo sin actualizar.** La sección 13 dice 16 procesos; se usan 10, y la sección 12 está vacía **[V]**.
- **Dato menor.** `espacios.json` justifica `max_features=0.2` con unas 26 columnas en Telco, y son 24 **[V]**. La raíz de 24 entre 24 es 0,204, así que el valor no cambia.
- **Memoria reportada.** El protocolo pide memoria de cada modelo. Solo se medirá en Telco con `memoria_por_modelo.csv`, con `tracemalloc`, que no ve toda la memoria nativa de XGBoost, LightGBM o CatBoost **[D]**.

## 10. Comandos, scripts y archivos

Raíz del proyecto: `C:\Users\natal\Desktop\Telco_Churn_Entregable`

### Archivos

| Archivo | Función |
|---|---|
| `experiments/protocolo.md` | Protocolo congelado |
| `experiments/espacios.json` | Espacios y presupuesto |
| `experiments/familia_holm.json` | Las 223 comparaciones |
| `experiments/confirmatorio/comun.py` | Semillas, datos, pipeline, modelos y métricas |
| `experiments/confirmatorio/correr_anidado.py` | Validación anidada |
| `experiments/confirmatorio/correr_con_reintentos.sh` | Envoltura con reinicio automático |
| `experiments/confirmatorio/robustez.py` | Robustez y sensibilidad |
| `experiments/confirmatorio/correr_finales.py` | Tenure y hold-out |
| `experiments/confirmatorio/complementos_confirmatorio.py` | Tablero, escalabilidad y SHAP |
| `experiments/confirmatorio/analizar.py` | Tablas y veredicto |
| `experiments/confirmatorio/estadistica.py` | Pruebas estadísticas |
| `experiments/confirmatorio/medir_tiempos.py` | Medición previa |
| `src/original_model.py` | `RegionalMAPEM` |

### Auxiliares de este informe, en el directorio temporal de la sesión

- `C:\Users\natal\AppData\Local\Temp\claude\fe17809bba669d90\1c52f1d0-11f9-41a2-bb22-76bda94d90a6\scratchpad\auditar_tiempos.py`
- `C:\Users\natal\AppData\Local\Temp\claude\fe17809bba669d90\1c52f1d0-11f9-41a2-bb22-76bda94d90a6\scratchpad\tiempos_tareas.csv`
- `C:\Users\natal\AppData\Local\Temp\claude\fe17809bba669d90\1c52f1d0-11f9-41a2-bb22-76bda94d90a6\scratchpad\tiempos_combinadores.csv`

### Inspección segura, solo lectura

Avance por conjunto, en Git Bash:

```bash
for d in experiments/resultados/crudos/*/; do echo "$d $(find "$d" -name '*.pkl' | wc -l)"; done
```

Últimas tareas:

```bash
tail -n 20 experiments/resultados/log_anidado.txt
```

Reinicios y errores:

```bash
grep -nE "=== intento|Traceback|WinError|^== " experiments/resultados/log_anidado.txt
```

Tiempos por modelo, sin métricas de desempeño:

```bash
/c/Users/natal/Downloads/anaconda/envs/ml_project/python.exe "C:/Users/natal/AppData/Local/Temp/claude/fe17809bba669d90/1c52f1d0-11f9-41a2-bb22-76bda94d90a6/scratchpad/auditar_tiempos.py" "C:/Users/natal/AppData/Local/Temp/claude/fe17809bba669d90/1c52f1d0-11f9-41a2-bb22-76bda94d90a6/scratchpad"
```

Cambios posteriores al tag:

```bash
git diff protocolo-v1 -- experiments/confirmatorio/correr_anidado.py
```

Frecuencia y carga por hilo, en PowerShell:

```powershell
Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | Where-Object Name -notmatch '_Total' | Select-Object Name,PercentProcessorTime,ProcessorFrequency,PercentProcessorPerformance
```

Memoria y paginación, en PowerShell:

```powershell
Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory | Select-Object AvailableMBytes,CommittedBytes,PagesInputPersec,PagesOutputPersec
```

### Reproducción. No ejecutar mientras siga la corrida principal

Competirían por la misma RAM y, sin `RESULTADOS_DIR`, por los mismos checkpoints.

Corrida vigente, ya en marcha:

```bash
PY=/c/Users/natal/Downloads/anaconda/envs/ml_project/python.exe bash experiments/confirmatorio/correr_con_reintentos.sh 10
```

Pasos siguientes, en orden, con `N` explícito y el `python` del entorno `ml_project`:

```bash
python experiments/confirmatorio/complementos_confirmatorio.py dashboard
```

```bash
python experiments/confirmatorio/robustez.py 10
```

```bash
python experiments/confirmatorio/correr_finales.py tenure 10
```

```bash
python experiments/confirmatorio/analizar.py
```

```bash
python experiments/confirmatorio/correr_finales.py holdout 10
```

```bash
python experiments/confirmatorio/analizar.py
```

```bash
python experiments/confirmatorio/complementos_confirmatorio.py escalabilidad
```

```bash
python experiments/confirmatorio/complementos_confirmatorio.py shap
```

Variables útiles: `RESULTADOS_DIR` cambia la carpeta de salida. `PRUEBA_N_TRIALS` reduce evaluaciones y sirve **solo** para pruebas de humo, nunca para resultados.

## Lo que no se pudo comprobar

- Si los 360 MHz de los hilos eficientes son reales y cuál es su causa: límite de potencia, temperatura, gestión de energía de Windows u OMEN.
- La temperatura y el perfil térmico activo.
- El pico de memoria por modelo y conjunto, en especial en KDDCup09.
- El costo en tiempo de la paginación.
- El determinismo real, repitiendo una tarea.
- Las exclusiones de Defender, que requieren permisos de administrador.
- Si la RAM del equipo se puede ampliar.
- Los tiempos reales de OpenML 40701, KDDCup09, combinadores fuera de Telco, escalabilidad, SHAP y `analizar.py`.
- Que `complementos_confirmatorio.py` y `analizar.py` funcionen de principio a fin: aún no se han ejecutado.
- El contenido actual de la guía, el correo del profesor y el libro: se resumió desde el historial.
