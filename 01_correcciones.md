# Trazabilidad de correcciones

Este capítulo responde, fila por fila, a las observaciones recibidas sobre el entregable anterior y a los criterios que
el docente fijó para el modelo original. Cada fila dice qué cambiamos, dónde está en el libro y con qué se puede
comprobar. Cuando una corrección alteró resultados, las tablas y pruebas que dependían de ellos se recalcularon.

## Observaciones del entregable anterior

La retroalimentación del entregable anterior (nota 4,2) reconoció el trabajo metodológico y señaló cinco descuentos. La
primera columna los cita tal como los escribió el docente.

| # | Observación del docente | Acción realizada | Ubicación en el libro | Evidencia |
|---|---|---|---|---|
| 1 | "no hay multifidelidad" | Successive Halving dentro de la validación anidada para Random Forest, XGBoost y SVM con SMOTE, comparado con una búsqueda aleatoria a fidelidad completa, con el mismo pipeline, las mismas semillas y los mismos pliegues internos | Modelamiento avanzado, sección 2.5 | `complementos/multifidelidad.py`; `resultados_modelado/complementarios/multifidelidad/` |
| 2 | "ni optimizaciones computacionales aplicadas ni mediciones de aceleración" | Optimizaciones aplicadas con aceleración medida, escalamiento en $n$ y $p$, paralelismo y perfilamiento | Modelamiento avanzado, secciones 2.4 y 2.6 | `complementos/optimizacion_computacional.py`; `resultados_modelado/complementarios/optimizacion_computacional/` |
| 3 | "no se reporta tamaño del efecto" | δ de Cliff, $d_z$ y correlación biserial de rangos en las comparaciones | Modelamiento avanzado, sección 3.1 | Tablas de comparaciones con su tamaño del efecto |
| 4 | "la calibración usa solo Platt" | Calibración isotónica de las 560 unidades de la realización principal, comparada con Platt | Modelamiento avanzado, secciones 1.7 y 1.8 | `complementos/calibracion_isotonica.py`; `resultados_modelado/complementarios/calibracion_isotonica/` |
| 5 | "Tampoco se entregan los resultados guardados ni `requirements.txt`, por lo que no se puede verificar la reproducción" | Resultados versionados, `requirements.txt` con versiones exactas, script que verifica la reproducción y apéndice con el código del entrenamiento | Modelamiento avanzado, sección 1.2; Apéndice: código del entrenamiento; Guía del repositorio | `resultados_modelado/`; `requirements.txt`; `verificar_reproduccion.py`; `CODIGO_EXPERIMENTO.ipynb` |

## Criterios del docente para el modelo original

El docente fijó estos criterios por correo; la primera columna los resume.

| # | Criterio del docente | Acción realizada | Ubicación en el libro | Evidencia |
|---|---|---|---|---|
| 6 | La novedad puede estar en la regla de combinación, si se diferencia de la mezcla de expertos con compuerta EM, la selección dinámica, el stacking con pesos locales y los ensambles ponderados | Revisión de literatura con búsqueda por citación y tabla de diferenciación de 11 trabajos en 7 familias | Capítulo 4, tabla 4.3 | `experiments/literatura/` |
| 7 | La demostración de convergencia debe enunciar sus supuestos | Proposición de concavidad, ascenso monótono y óptimo global, con los supuestos S1 a S4 y el caso de borde | Modelo propio, sección 5 | `tests/test_original_model.py` verifica el ascenso monótono |
| 8 | El log-loss vale como métrica principal si se pre-registra declarando que se eligió después de los pilotos | Log-loss fijado en el protocolo con esa declaración, antes del tag | Capítulo 6, tabla del protocolo | `experiments/protocolo.md`, sección 3; tag `protocolo-v1` |
| 9 | Los benchmarks deben tener el mismo presupuesto y calibración (Platt o isotónica) dentro del bucle interno | 30 evaluaciones de TPE para todos, con Platt dentro del bucle interno y sin SMOTE; presupuesto consumido reportado | Capítulo 6, tabla 6.1 | `experiments/espacios.json`; `tablas/costos_por_pliegue.csv` |
| 10 | F1 y AUC deben reportarse y contar en la corrección por comparaciones múltiples | Ambas métricas, con Nadeau-Bengio y Wilcoxon, dentro de la familia de Holm de 223 comparaciones | Capítulo 6, tablas 6.4 y 6.7 | `experiments/familia_holm.json`, bloque B |
| 11 | Pruebas admisibles: Wilcoxon o Nadeau-Bengio, permutación pareada, Friedman con Nemenyi y δ de Cliff de al menos 0,147 | Las cuatro pruebas, con Holm, intervalos BCa y δ de Cliff; además DeLong y McNemar en el hold-out | Capítulo 6 | `tablas/familia_resultados.csv` |
| 12 | El SVM+ con Churn Score se descarta por riesgo de fuga | Descartado; las variables privilegiadas no pertenecen al dataset del proyecto | Modelo propio, sección 10; protocolo, sección 11 | Piloto 1 del registro de iteraciones |
| 13 | Un resultado negativo honesto tiene tope de 4,0 | R3 se reporta como no alcanzado, con su causa probable | Capítulos 6 y 9 | `tablas/veredicto_R3.json` |

## Auditoría de fuga de información

| Fuente posible de fuga | Control |
|---|---|
| Preprocesamiento ajustado con todos los datos | El ETL (`HistoricalFeatures`) solo transforma fila por fila; el escalado y la codificación viven dentro del `Pipeline` y se ajustan en cada pliegue de entrenamiento, también en la calibración y en la búsqueda interna |
| Variables que codifican la respuesta | El dataset del proyecto no incluye `Churn Score`, `CLTV` ni `Churn Reason`; por eso se descartó el SVM+ que las usaba |
| Duplicados entre particiones | Telco no repite clientes (`customerID` es único), pero 73 clientes comparten todos sus atributos con algún otro, y 13 del hold-out son idénticos a alguno de desarrollo. Todos tienen un mes de antigüedad, contrato mes a mes y, en dos de cada tres casos, sin internet: son clientes distintos con el perfil más simple posible, no registros copiados, así que se conservan; quitarlos cambiaría la población que se quiere predecir. En Iranian, que no trae identificador, se quitaron las 300 filas repetidas antes de partir |
| Identificadores | `customerID` y `phone_number` se eliminan antes de modelar |
| Hold-out | Reservado desde el entregable anterior (`test_index.csv`). Los pilotos 3 a 5 lo usaron por error con la versión extendida del dataset; sus cifras se descartaron y la exposición quedó declarada en la sección 4.1 del protocolo. La evaluación confirmatoria del hold-out se hizo una sola vez, después de sellar el análisis |
| Mezcla temporal | Telco es de corte transversal; el único ordenamiento con sentido, la antigüedad, se usó para un estudio explícito de cambio de distribución (capítulo 7) |

## Cambios en el ranking

Con F1 como criterio y SMOTE, el entregable anterior ponía arriba a Random Forest y XGBoost. Con el log-loss como métrica
principal, todos los modelos calibrados y sin remuestreo, el mejor benchmark pasa a ser la logística con splines, y
XGBoost y Random Forest caen a los puestos 11 y 13 entre los 17 modelos de Telco. No es una contradicción: F1 premia
el umbral de decisión y el log-loss premia la probabilidad. La tabla 7.16 muestra las combinaciones del entregable
anterior evaluadas en log-loss sobre los mismos pliegues: compiten en F1, pero sus probabilidades están mal calibradas.
