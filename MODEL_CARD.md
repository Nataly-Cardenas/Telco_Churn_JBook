# Ficha del modelo: RegionalMAPEM

## Descripción

RegionalMAPEM combina las probabilidades de 11 clasificadores calibrados (los benchmarks) con pesos que dependen de la
región del espacio de clientes:

$$p(\text{churn} \mid x) = \sum_{k=1}^{K} r_k(x) \sum_{m=1}^{M} \pi_{km}\, p_m(x), \qquad \pi_k \sim \mathrm{Dirichlet}(1 + c\,g).$$

Las regiones $r_k(x)$ salen de un GMM sobre dos componentes principales de las variables continuas; los pesos
$\pi_{km}$ se estiman por máxima posterior con EM, con un prior que los encoge hacia los pesos globales $g$.

| Campo | Valor |
|---|---|
| Autoras | Laura S. Rivera Figueredo y Natalý A. Cárdenas Izaquita (Universidad del Norte) |
| Versión | `protocolo-v1` (protocolo congelado el 2026-11-08) |
| Código | `src/original_model.py`, clase `RegionalMAPEM` (interfaz de scikit-learn) |
| Pruebas | `tests/test_original_model.py` (12 pruebas) |
| Hiperparámetros propios | $K \in [1, 8]$, $c \in [0{,}01;\,200]$ (escala log), covarianza `diag` o `full` |
| Tarea | Clasificación binaria con salida probabilística (abandono en el último mes) |

## Uso previsto

Ordenar a los clientes por riesgo de abandono para priorizar acciones de retención, con probabilidades calibradas. No
está pensado para decidir de forma automática sobre un cliente individual sin revisión humana, ni para explicar causas
del abandono: las variables describen el estado del cliente y sus asociaciones no son causales.

## Datos de entrenamiento y evaluación

Telco Customer Churn (IBM, versión de Kaggle): 5634 clientes de desarrollo y 1409 de hold-out, partición estratificada
fija del entregable anterior. Evaluación adicional en Iranian Churn (UCI 563), churn de OpenML 40701 y KDDCup09 (OpenML
1112). Detalles en `DATA_STATEMENT.md`.

## Desempeño

| Evaluación | Log-loss | F1 | AUC | ECE |
|---|---|---|---|---|
| Telco, validación anidada 5 × 5 (media) | 0,4113 | 0,634 | 0,850 | 0,025 |
| Telco, hold-out (1409 clientes, una sola evaluación) | 0,4109 | 0,636 | 0,850 | 0,011 |
| Mejor benchmark en Telco (logística con splines), validación anidada | 0,4113 | 0,638 | 0,850 | 0,024 |

No es superior al mejor benchmark: la diferencia en Telco (−0,00003 de log-loss) no es significativa con ninguna de las
pruebas del protocolo (capítulo 6 del libro). En los conjuntos adicionales es el mejor en media solo en Iranian.

## Limitaciones

- Necesita a los 11 expertos para predecir: unos 23 MB serializados y el costo de inferencia de Random Forest y SVM.
- En Telco las regiones no aportan frente a los pesos globales; el modelo se comporta como un ensamble ponderado.
- Las probabilidades se evaluaron en un solo operador y una sola época; con otra población hay que volver a validarlo.
- Al 90 % de cobertura, un tercio de los clientes recibe un conjunto conforme con las dos clases: el modelo no tiene
  certeza sobre ellos.

## Consideraciones éticas

Las variables incluyen `gender` y `SeniorCitizen`. Su peso SHAP es bajo, pero no se hizo un análisis formal de equidad
entre grupos. Un uso inadecuado sería negar beneficios o cambiar condiciones de un cliente solo por su riesgo predicho,
sin considerar el costo de los falsos positivos.
