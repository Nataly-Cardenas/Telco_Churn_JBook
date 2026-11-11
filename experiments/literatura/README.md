# Búsqueda por citación y en índices abiertos (2026-11-08)

Complementa la revisión en WoS, Scopus, Scopus AI y Google Académico del 2026-11-03 y 2026-11-07 (Sheet
*Candidatos_a_investigar*). Todo se reproduce con `busqueda_citas.py`, que consulta OpenAlex y arXiv.

| Archivo | Contenido |
|---|---|
| `snowballing_resumen.csv` | Por cada semilla: citas hacia adelante, referencias hacia atrás y cuántas tratan de churn |
| `snowballing_churn.csv` | Los trabajos encontrados que mencionan churn o abandono, con título, año, DOI y resumen |
| `matriz_busqueda_abierta.csv` | Cadenas de búsqueda con resultados, revisados, candidatos y fecha |

## Semillas

De Caigny et al. (2018), Soares et al. (2006), Yan et al. (2001), Sill et al. (2009) y Jordan y Jacobs
(1994). Hacia adelante solo se descargan los trabajos con *churn*, *attrition* o *retention* en título o
resumen. Se revisaron título y resumen de los que además mencionan clústeres, regiones, compuertas, EM o
Dirichlet.

## Qué cambió

**Antecedente nuevo y cercano.** Dos artículos en chino sobre churn en telecomunicaciones, encontrados con la
cadena "dynamic classifier selection" AND churn:

- *Telecommunication customer churn prediction based on the dynamic integration of multiple classifiers*
  (2010). Agrupa con k-means en subáreas, entrena Naive Bayes, perceptrón multicapa y J48 en cada subárea y
  los integra linealmente con pesos optimizados por colonia de abejas.
- *A Study on Prediction of Telecom Customer Churn Based on Dynamic Selection, Optimization and Integration of
  Cost Sensitivity* (2012). El mismo esquema, con función de beneficio y cardumen artificial.

La idea de integrar clasificadores por zona del espacio de clientes **ya existe en churn**. Lo que no existe en
ellos es lo que define nuestra regla: pertenencia suave, expertos fijos que no se reentrenan por región, MAP-EM
con objetivo cóncavo y garantía de óptimo global, y el prior de Dirichlet hacia los pesos globales. La
declaración de novedad se limita a eso.

**Otros cercanos revisados por resumen.**

- ClusGBDT (2020): gradient boosting por clúster, sin pesos entre expertos.
- MOEEC (*PLOS ONE*, 2024): ensamble evolutivo basado en clústeres, multiobjetivo, para elegir clasificadores
  diversos.
- *Customer churn model based on complementarity measure and random forest* (2021): poda de árboles con clústeres.

**Sin antecedente en churn.** La familia FWLS y stacking con pesos locales (dos citas de Sill 2009 con churn o
retención; ninguna usa pesos por región). Mezcla de expertos con churn en arXiv: los dos resultados son modelos de
lenguaje, donde "churn" significa otra cosa.

## Lo que este índice no reemplaza

OpenAlex no cubre todo lo que cubren WoS y Scopus, y varios resúmenes faltan. Quedan para el equipo, con acceso
institucional:

1. Repetir el "Citado por" de Google Académico y el índice de citas de WoS sobre las mismas semillas.
2. Leer a texto completo los dos artículos de 2010 y 2012 (confirmar si los pesos cambian por subárea o son
   globales), *Segment-Aware Learning for Adaptive Customer Churn Prediction* (IEEE BigData 2025) y *Hybrid
   black-box classification for customer churn prediction with segmented interpretability analysis* (*Decision
   Support Systems*, 2024), que OpenAlex lista sin resumen.
3. Registrar las cadenas de stacking con pesos locales en WoS, Scopus y Google Académico, con fecha.
4. Repetir la búsqueda justo antes de la entrega, como pide el protocolo de revisión.
