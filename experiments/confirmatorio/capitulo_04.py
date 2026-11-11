"""Capítulo 4 del libro: revisión de literatura y novedad."""
from construir_capitulos import code, md

ARCHIVO = "04_literatura.ipynb"

CELDAS = [
md(r"""
# Revisión de literatura y novedad

Antes de afirmar que un modelo es nuevo hay que buscar con cuidado a quién se le ocurrió primero. Este capítulo
documenta esa búsqueda: dónde buscamos, con qué cadenas, qué encontramos y en qué se diferencia RegionalMAPEM de los
trabajos más cercanos. La conclusión adelantada es que la idea general (combinar clasificadores según la zona del
espacio de clientes) ya existe en churn, y que lo nuevo es la regla con la que nosotros los combinamos.

## Cómo buscamos

| Etapa | Fecha | Fuentes | Qué se hizo |
|---|---|---|---|
| Búsqueda por palabras clave | 2026-11-03 | Web of Science, Scopus, Scopus AI y Google Académico | Cadenas que combinan la familia del modelo (mezcla de expertos, ensamble ponderado, selección dinámica, pesos por región), el problema (churn, abandono, attrition) y sinónimos en inglés y español |
| Ampliación por familias afines | 2026-11-07 | Las mismas bases | Modelos por segmento (logit leaf model), DES-Clustering, combinación de pronósticos por regímenes y stacking con pesos locales |
| Índices abiertos y citación | 2026-11-08 | OpenAlex y arXiv | 17 cadenas registradas con sus conteos; búsqueda por citación hacia atrás y hacia adelante desde cinco trabajos semilla |

La búsqueda por citación se hizo con OpenAlex porque permite descargar de forma reproducible las referencias y las
citas de un trabajo (`experiments/literatura/busqueda_citas.py`). Hacia adelante solo se conservaron los trabajos con
*churn*, *attrition* o *retention* en el título o el resumen, y de ellos se leyeron completos los resúmenes que además
mencionan clústeres, regiones, compuertas, EM o Dirichlet.
"""),
code(r"""
# Matriz de búsqueda en índices abiertos (OpenAlex y arXiv, 2026-11-08)
import sys
from pathlib import Path
import pandas as pd
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from src.presentacion import tabla
LIT = ROOT / "experiments" / "literatura"
M = pd.read_csv(LIT / "matriz_busqueda_abierta.csv")
tabla(pd.DataFrame({"Base": M.base, "Cadena": M.cadena, "Resultados": M.resultados, "Revisados (título y resumen)": M.revisados,
                    "Relevantes": M.candidatos, "Trabajos relevantes": M.titulos_candidatos.fillna("—").str.replace(" | ", "; ", regex=False),
                    "Fecha": M.fecha}),
      "Tabla 4.1. Matriz de búsqueda en índices abiertos", decimales=0,
      nota="En las cadenas con más de 25 resultados se revisaron los 25 primeros por relevancia.")
"""),
md(r"""
La mayoría de los resultados relevantes no tratan de combinar clasificadores por regiones: son segmentaciones de
clientes seguidas de un modelo por segmento, o ensambles con pesos globales. Las dos excepciones aparecen con la cadena
de selección dinámica: dos artículos sobre churn en telecomunicaciones, de 2010 y 2012, que se discuten abajo.
"""),
code(r"""
# Búsqueda por citación desde los trabajos semilla
S = pd.read_csv(LIT / "snowballing_resumen.csv")
tabla(pd.DataFrame({"Trabajo semilla": S.semilla, "Año": S.anio, "Citas hacia adelante": S.citas_adelante,
                    "Referencias hacia atrás": S.referencias_atras, "Trabajos sobre churn entre las citas": S.con_churn, "Fecha": S.fecha}),
      "Tabla 4.2. Búsqueda por citación (OpenAlex)", decimales=0, buscar=False)
C = pd.read_csv(LIT / "snowballing_churn.csv")
print(f"Trabajos sobre churn encontrados por citación: {len(C)}")
"""),
md(r"""
## Los trabajos más cercanos

Comparar solo por aplicación dejaría por fuera a los métodos más parecidos, así que comparamos por familia algorítmica. La
tabla 4.3 reúne once trabajos de siete familias y precisa, para cada uno, qué conserva nuestra propuesta y qué cambia.

| Trabajo | Familia | ¿Churn? | Qué conserva la propuesta | Qué cambia |
|---|---|---|---|---|
| De Caigny, Coussement y De Bock (2018), logit leaf model {cite}`decaigny2018` | Modelos por segmento (árbol + logística por hoja) | Sí | Tratar distinto a cada segmento de clientes | Las hojas son segmentos duros de un árbol supervisado y en cada uno se entrena un modelo nuevo. Aquí las regiones son suaves y no supervisadas, y no se entrena nada nuevo: se reponderan benchmarks fijos con pesos MAP encogidos hacia los globales |
| Extensiones del logit leaf model (De Caigny et al., 2021; Hendriks, 2022; Idbenjra et al., 2024) | Modelos por segmento | Sí (las dos primeras) | Ídem | Un modelo por hoja; ninguna combina clasificadores con pesos por segmento |
| Soares et al. (2006), DES-Clustering {cite}`soares2006` | Selección dinámica por regiones de k-means | No encontrado en churn | Regiones obtenidas por clustering | Se elige un subconjunto de clasificadores por precisión y diversidad (selección dura). Aquí se estiman pesos continuos por máxima posterior |
| Integración dinámica por subáreas (2010, 2012; artículos en chino sobre telecomunicaciones) | k-means + clasificadores por subárea + integración lineal | Sí | Combinar clasificadores según la zona del espacio de clientes | Regiones duras; un clasificador nuevo por subárea; pesos por metaheurística (colonia de abejas, cardumen) sin garantía de óptimo. Aquí las regiones son suaves, los expertos fijos y los pesos salen de MAP-EM con objetivo cóncavo y prior hacia los globales |
| Xiao et al. (2012, 2014) {cite}`xiao2012` | Selección dinámica | Sí | La competencia de cada modelo varía en el espacio de entrada | Selección dura frente a pesos suaves con prior de Dirichlet |
| Bilal et al. (2022) | Clustering (K-medoids) + ensambles globales | Sí (telecomunicaciones) | Combinar clustering y clasificadores | Votación, stacking y AdaBoost globales; no hay pesos por clúster ni EM |
| Elliott y Timmermann (2005) {cite}`elliott2005` | Combinación de pronósticos por regímenes | No (macroeconomía) | Pesos de combinación que cambian según un régimen | Regímenes temporales (cadena de Markov latente) y pronósticos puntuales con pérdida cuadrática, frente a regiones del espacio de clientes y probabilidades con log-loss y prior de Dirichlet |
| Jacobs et al. (1991); Jordan y Jacobs (1994); Yan et al. (2001) {cite}`jordan1994` | Mezcla de expertos | Yan (2001): sí, neuronal | Mezcla convexa con pesos que dependen de $x$, estimada por EM | La compuerta se fija antes de estimar los pesos y los expertos no se reentrenan. Por eso el objetivo es cóncavo y el EM llega al óptimo global; en la mezcla de expertos clásica, compuerta y expertos se entrenan juntos y el EM solo garantiza un óptimo local |
| Jiang et al. (2024); Khoh et al. (2023) | Ensamble ponderado global | Sí | Combinación ponderada de clasificadores | Un solo vector de pesos, ajustado por metaheurística o por desempeño |
| Sill, Takács, Mackey y Lin (2009), FWLS {cite}`sill2009` | Stacking con pesos locales | No encontrado en churn | Pesos de combinación que dependen de $x$ | En FWLS los pesos son funciones lineales de metavariables, sin restricción de símplex, y se ajustan por mínimos cuadrados regularizados. Aquí son pesos convexos por región, con verosimilitud binaria, prior de Dirichlet y óptimo global garantizado |
| Arifin y Wibowo (2025), LESTARI | Stacking con pesos por instancia | No (compras prepago) | Pesos que varían por cliente | Regresores sin modelo probabilístico ni garantías |

**Tabla 4.3.** Diferenciación frente a métodos afines.

Los dos trabajos de 2010 y 2012 son los antecedentes más cercanos: agrupan a los clientes con k-means, entrenan
clasificadores en cada subárea y los integran con pesos optimizados. Comparten con nuestra propuesta la intuición de
que la mejor combinación depende de la zona del espacio de clientes. Difieren en el mecanismo: regiones duras,
clasificadores reentrenados por subárea y pesos hallados por metaheurísticas, sin garantía de óptimo ni un prior que
proteja a las regiones con pocos datos.

## Declaración de novedad

La novedad está en la regla de combinación, no en segmentar clientes ni en integrar clasificadores por zonas, porque
ambas cosas ya existen en churn.

> Al mejor de nuestro conocimiento y según la búsqueda descrita (Web of Science, Scopus, Scopus AI y Google Académico,
> 2026-11-03 y 2026-11-07; OpenAlex y arXiv, con búsqueda por citación desde cinco trabajos semilla, 2026-11-08), no se
> encontraron estudios de predicción de churn que combinen clasificadores fijos y calibrados con pesos que dependen de
> regiones suaves del espacio de clientes, estimados por máxima posterior con EM y un prior de Dirichlet hacia los pesos
> globales.

Esa regla tiene una consecuencia que no comparte ninguno de los métodos de la tabla: con las regiones fijas, el
objetivo es cóncavo sobre un producto de símplex, así que el EM sube en cada iteración y converge al óptimo global
(capítulo 5). Los resultados de los capítulos 6 y 7 muestran que esa garantía no se traduce en una ventaja predictiva
en Telco, pero la novedad del mecanismo no depende de que gane.

## Alcance de la búsqueda

OpenAlex no cubre todo lo que cubren Web of Science y Scopus, y varios resúmenes no están disponibles. Por eso la
búsqueda por citación en índices abiertos complementa, y no reemplaza, la de las bases institucionales. Los dos
artículos de 2010 y 2012 se analizaron a partir de sus resúmenes; la diferenciación de la tabla 4.3 se apoya en lo que
esos resúmenes describen (regiones de k-means, clasificadores por subárea y pesos por metaheurística).
"""),
]
