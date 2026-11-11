# Declaración de los datos

## Conjunto principal: Telco Customer Churn

| Campo | Valor |
|---|---|
| Procedencia | IBM, publicado como conjunto de muestra; versión de Kaggle (`blastchar/telco-customer-churn`), archivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Tamaño | 7043 clientes × 21 columnas (19 predictoras, el identificador `customerID` y la respuesta `Churn`) |
| Naturaleza | Datos de una compañía de telecomunicaciones ficticia creados por IBM con fines de demostración; un registro por cliente, de corte transversal |
| Respuesta | `Churn`: el cliente se dio de baja en el último mes (26,5 %) |
| Partición | Desarrollo de 5634 clientes (`train_index.csv`) y hold-out de 1409 (`test_index.csv`), estratificada con semilla 42 |
| Faltantes | 11 valores vacíos de `TotalCharges`, todos de clientes con `tenure = 0`; se reemplazan por 0 |
| Licencia | La de la publicación en Kaggle; los datos de muestra de IBM se distribuyen para uso educativo y de demostración |

**Sesgos y limitaciones conocidos.** Es un solo operador ficticio y una sola fotografía en el tiempo, así que no
representa a ningún mercado real. 73 clientes comparten todos sus atributos con algún otro (perfiles de un mes de
antigüedad y contrato mes a mes); son clientes distintos y se conservan. Las variables demográficas se limitan a
`gender`, `SeniorCitizen`, `Partner` y `Dependents`. La versión extendida de IBM, con `Churn Score`, `CLTV` y
`Churn Reason`, no se usa: esas variables codifican la respuesta.

## Conjuntos adicionales

| Conjunto | Fuente y licencia | n | p | % churn | Tratamiento |
|---|---|---|---|---|---|
| Iranian Churn | UCI Machine Learning Repository, id 563, CC BY 4.0 | 3150 → 2850 | 13 | 15,7 | Se quitan las 300 filas repetidas (no hay identificador) |
| churn (telecom EE. UU.) | OpenML 40701, licencia "public" | 5000 | 19 | 14,1 | Se quita `phone_number` (identificador); `state` y `area_code` se tratan como categorías |
| KDDCup09_churn (Orange) | OpenML 1112, licencia "Public" | 50 000 → 10 000 | 230 | 7,3 | Submuestra estratificada fija (`seed_for('subset', 1112)`); 70 % de celdas faltantes |

Los conjuntos adicionales se eligieron antes de ver el desempeño del modelo en ellos, con el criterio fijado en la
sección 4 de `experiments/protocolo.md`. Los archivos usados están en `experiments/datos/`.
