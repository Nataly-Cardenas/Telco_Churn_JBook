"""Capítulo 4 del libro: revisión de literatura y novedad."""
from construir_capitulos import code, md

ARCHIVO = "04_literatura.ipynb"

CELDAS = [
md(r"""
# Revisión de literatura y novedad

Antes de afirmar que un modelo es nuevo hay que buscar con cuidado a quién se le ocurrió primero. Este capítulo
documenta esa búsqueda: dónde buscamos, con qué cadenas, qué encontramos y en qué se diferencia RegionalMAPEM de los
trabajos más cercanos. La conclusión adelantada es que la idea general (combinar clasificadores según la zona del
espacio de clientes) ya existe en churn, y que lo nuevo es la regla con la que nosotras los combinamos.

## Cómo buscamos

| Etapa | Fecha | Fuentes | Qué se hizo |
|---|---|---|---|
| Mapa de modelos usados en churn | 2026-10-30 | Web of Science, Scopus, IEEE Xplore y arXiv | Dos conjuntos por base: A, clasificación con ML con *novel*, *improved*, *enhanced* o *hybrid* en el título, y B, churn con ML, ambos sin redes neuronales; además B2, churn amplio, como control. En Web of Science y Scopus suman 27 313 registros. De ahí salieron ocho modelos candidatos, verificados después en Google Académico y arXiv |
| Búsqueda del mecanismo propio | 2026-11-03 | Web of Science, Scopus, Scopus AI y Google Académico | Cadenas que combinan la definición de churn con cada familia afín: mezcla de expertos, selección dinámica, stacking local y ensambles ponderados, EM y agregación de probabilidades, y calibración. También se repitieron A, B y B2 en Web of Science |
| Índices abiertos y citación | 2026-11-08 | OpenAlex y arXiv | Cadenas adicionales, una de ellas en español, y búsqueda por citación hacia atrás y hacia adelante desde cinco trabajos semilla |
| Repositorios de código | 2026-11-10 | GitHub | Búsqueda por nombre, descripción, README y contenido del código, con lectura de los repositorios más cercanos |

La definición de churn combina el término en inglés con sus sinónimos en inglés y, en Web of Science, en español: *customer
defection*, *customer attrition*, "fuga de clientes", "abandono de clientes" y "deserción de clientes". Las cadenas exactas,
con sus resultados, están en la tabla 4.4.

## Matriz de novedad

La tabla 4.1 resume la búsqueda del mecanismo propio. Las filas de las bases institucionales vienen del registro del 3 de
noviembre, y las de OpenAlex y arXiv, del 8 de noviembre. En las cadenas con muchos resultados se revisó solo una parte: la
columna de revisados dice cuántos.
"""),
code(r"""
# Matriz de novedad del modelo propio: registro institucional (2026-11-03) e índices abiertos (2026-11-08)
import sys
from pathlib import Path
import pandas as pd
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
from src.presentacion import tabla
LIT = ROOT / "experiments" / "literatura"
leer = lambda n: pd.read_csv(LIT / n, dtype=str, keep_default_na=False)   # los conteos se muestran tal como se registraron

N = leer("matriz_novedad.csv")
PROPIO = N[N["Modelo candidato"].str.contains("propio") & (N["Base de datos"] != "Declaración")]
A = leer("matriz_busqueda_abierta.csv")
filas = pd.concat([
    PROPIO[["Fecha", "Base de datos", "Cadena de búsqueda", "Resultados", "Revisados", "Relevantes", "Observación"]],
    pd.DataFrame({"Fecha": A.fecha, "Base de datos": A.base, "Cadena de búsqueda": A.cadena, "Resultados": A.resultados, "Revisados": A.revisados,
                  "Relevantes": A.candidatos, "Observación": A.titulos_candidatos.replace("", "Sin trabajos relevantes").str.replace(" | ", "; ", regex=False)})])
tabla(filas, "Tabla 4.1. Matriz de novedad del mecanismo propio", decimales=0,
      nota="[churn] es la definición de la tabla 4.4 (fila NOTA-1103). Los identificadores entre paréntesis remiten a esa tabla.")
"""),
md(r"""
Ninguna fila contiene el mecanismo completo. En las bases institucionales, lo más cercano es la mezcla de expertos neuronal de
Yan et al. (2001), la selección dinámica de Xiao et al. (2012 y 2014), y una serie de ensambles con pesos globales, entre
ellos el de Khoh et al. (2023). Cuando el EM aparece junto a churn, se usa para segmentar o en modelos de tópicos, no para
combinar clasificadores. Scopus AI, consultado en lenguaje natural, tampoco devolvió estudios con pesos locales estimados
con EM o MAP-EM.

Los antecedentes que obligan a acotar la novedad aparecieron en la cadena de selección dinámica de OpenAlex: dos artículos
sobre churn en telecomunicaciones, de 2010 y 2012, que se discuten abajo.
"""),
code(r"""
# Búsqueda por citación desde los trabajos semilla
S = leer("snowballing_resumen.csv")
tabla(pd.DataFrame({"Trabajo semilla": S.semilla, "Año": S.anio, "Citas hacia adelante": S.citas_adelante,
                    "Referencias hacia atrás": S.referencias_atras, "Trabajos sobre churn entre las citas": S.con_churn, "Fecha": S.fecha}),
      "Tabla 4.2. Búsqueda por citación (OpenAlex)", decimales=0, buscar=False)
"""),
md(r"""
La búsqueda por citación se hizo con OpenAlex porque permite descargar de forma reproducible las referencias y las citas de
cada trabajo (`experiments/literatura/busqueda_citas.py`). Hacia adelante solo se conservaron los trabajos con *churn*,
*attrition* o *retention* en el título o el resumen.

## Los trabajos más cercanos

Comparar solo por aplicación dejaría por fuera a los métodos más parecidos, así que comparamos por familia algorítmica. La
tabla 4.3 reúne diez familias y precisa, para cada una, qué conserva nuestra propuesta y qué cambia.

| Familia y trabajos | ¿Churn? | Qué conserva la propuesta | Qué cambia |
|---|---|---|---|
| **Mezcla de expertos con compuerta EM:** Jacobs et al. (1991); Jordan y Jacobs (1994) {cite}`jordan1994`; Yan et al. (2001); LRMoE, Fung et al. (2019) | Sí, pero neuronal (Yan, 2001; MMoE de Ziadi et al.). LRMoE se usa en tarificación de seguros, no para predecir churn | Mezcla convexa con pesos que dependen de $x$, estimada por EM | La compuerta se fija antes de estimar los pesos y los expertos no se reentrenan. Por eso el objetivo es cóncavo y el EM llega al óptimo global; cuando compuerta y expertos se entrenan juntos, el EM solo garantiza un óptimo local |
| **Selección dinámica:** Woods et al. (1997); Ko et al. (2008); Cruz et al. (2015); DES-Clustering, Soares et al. (2006) {cite}`soares2006`; Xiao et al. (2012, 2014) {cite}`xiao2012` | Sí (Xiao et al.) | La competencia de cada modelo varía en el espacio de entrada; DES-Clustering obtiene regiones por clustering | Selección dura de clasificadores frente a pesos continuos estimados por máxima posterior con un prior de Dirichlet |
| **Integración dinámica por subáreas** (2010 y 2012; dos artículos en chino sobre telecomunicaciones) | Sí | Combinar clasificadores según la zona del espacio de clientes | Regiones duras de k-means, un clasificador nuevo por subárea y pesos por metaheurística sin garantía de óptimo. Aquí las regiones son suaves, los expertos fijos y los pesos salen de MAP-EM con objetivo cóncavo |
| **Modelos por segmento:** logit leaf model, De Caigny et al. (2018) {cite}`decaigny2018`; extensiones (De Caigny et al., 2021; Hendriks, 2022; Idbenjra et al., 2024) | Sí | Tratar distinto a cada segmento de clientes | Segmentos duros de un árbol supervisado, con un modelo nuevo por hoja. Aquí no se entrena nada nuevo: se reponderan benchmarks fijos |
| **Clustering con ensambles globales:** Bilal et al. (2022) | Sí | Combinar clustering y clasificadores | Votación, stacking y AdaBoost globales, sin pesos por clúster ni EM |
| **Stacking y ensambles con pesos globales:** Wolpert (1992); Breiman (1996); Khoh et al. (2023); Jiang et al. (2024) | Sí | Usar predicciones fuera de pliegue para ajustar una combinación ponderada | Un solo vector de pesos, frente a pesos por región con un prior hacia los globales |
| **Stacking con pesos locales:** FWLS, Sill et al. (2009) {cite}`sill2009`; LESTARI, Arifin y Wibowo (2025) | No: LESTARI predice compras de datos prepago | Pesos que varían con la instancia | En FWLS los pesos son lineales en metavariables, sin restricción de símplex, y se ajustan por mínimos cuadrados. Aquí son pesos convexos por región con verosimilitud binaria, prior de Dirichlet y óptimo global |
| **Agregación de probabilidades:** pool lineal, Ranjan y Gneiting (2010); Gneiting y Ranjan (2013) | No encontrado | Combinación lineal de probabilidades | Pesos globales frente a pesos por región aprendidos con MAP-EM |
| **Selección de ensambles centrada en la decisión:** Baumann et al. (2015) | Sí | Combinar modelos optimizando la métrica relevante | Selección global y discreta de miembros, frente a combinación suave por regiones con objetivo de log-verosimilitud |
| **Combinación por regímenes:** Elliott y Timmermann (2005) {cite}`elliott2005` | No (macroeconomía) | Pesos de combinación que cambian según un régimen | Regímenes temporales y pronósticos puntuales con pérdida cuadrática, frente a regiones del espacio de clientes y probabilidades |

**Tabla 4.3.** Diferenciación frente a métodos afines.

Los dos trabajos de 2010 y 2012 son los antecedentes más cercanos. Comparten con nuestra propuesta la intuición de que la
mejor combinación depende de la zona del espacio de clientes. Difieren en el mecanismo: regiones duras, clasificadores
reentrenados por subárea y pesos hallados por metaheurísticas, sin un prior que proteja a las regiones con pocos datos.

## Declaración de novedad

La novedad está en la regla de combinación, no en segmentar clientes ni en integrar clasificadores por zonas, porque
ambas cosas ya existen en churn.

> Al mejor de nuestro conocimiento y según la búsqueda descrita (Web of Science, Scopus, Scopus AI y Google Académico,
> 2026-11-03; OpenAlex y arXiv, con búsqueda por citación desde cinco trabajos semilla, 2026-11-08), no se encontraron
> estudios de predicción de churn que combinen clasificadores fijos y calibrados con pesos que dependen de regiones suaves
> del espacio de clientes, estimados por máxima posterior con EM y un prior de Dirichlet hacia los pesos globales.

Esa regla tiene una consecuencia que no comparte ninguno de los métodos de la tabla: con las regiones fijas, el objetivo es
cóncavo sobre un producto de símplex, así que el EM sube en cada iteración y converge al óptimo global (capítulo 5). Los
resultados de los capítulos 6 y 7 muestran que esa garantía no se traduce en una ventaja predictiva en Telco, pero la novedad
del mecanismo no depende de que gane.

## Alcance de la búsqueda

La revisión se apoya en títulos y resúmenes. Los dos artículos de 2010 y 2012 se analizaron a partir de sus resúmenes, y la
diferenciación de la tabla 4.3 se apoya en lo que esos resúmenes describen. La búsqueda por citación se hizo en OpenAlex, que
no cubre todo lo que cubren Web of Science y Scopus. La ampliación hacia modelos por segmento, DES-Clustering y combinación
por regímenes se hizo por familia de métodos: esos trabajos están en la tabla 4.3, pero sus búsquedas no tienen conteos
registrados y por eso no aparecen en la matriz.

## Registro completo de búsquedas

La tabla 4.4 conserva las cadenas exactas, con la fecha y el número de resultados. La tabla 4.5 muestra la verificación de
los ocho modelos que la primera etapa dejó como candidatos: todos quedaron descartados al buscar en Google Académico, lo que
llevó a proponer un mecanismo propio en lugar de aplicar a churn un modelo existente.
"""),
code(r"""
# Registro de búsquedas: cadenas exactas, tal como se ejecutaron
B = leer("registro_busquedas.csv")
tabla(B[["ID", "Base", "Conjunto", "Cadena exacta", "Fecha", "Resultados", "Registros exportados", "Estado", "Observación"]],
      "Tabla 4.4. Registro de búsquedas en bases institucionales y arXiv", decimales=0,
      nota="Las filas con sufijo 1103 son del 2026-11-03. NOTA-1103 define [churn].")

# Verificación de los ocho candidatos de la primera etapa
C = N[~N["Modelo candidato"].str.contains("propio")]
tabla(C[["Fecha", "Base de datos", "Modelo candidato", "Cadena de búsqueda", "Resultados", "Revisados", "Relevantes", "Observación"]],
      "Tabla 4.5. Verificación de los modelos candidatos de la primera etapa", decimales=0)
"""),
md(r"""
## Repositorios de código afines

Un mecanismo puede estar implementado sin haberse publicado, así que también revisamos GitHub. Las búsquedas cubren el nombre,
la descripción, el README y el contenido de los archivos de Python, y se hacen con y sin la palabra churn.
"""),
code(r"""
# Búsqueda en GitHub y repositorios más cercanos, revisados uno por uno
RC, RR, RV = leer("repositorios_cadenas.csv"), leer("repositorios_resultados.csv"), leer("repositorios_revision.csv")
tabla(pd.DataFrame({"Fecha": RC.fecha, "Búsqueda": RC.base, "Cadena": RC.cadena, "Resultados": RC.resultados, "Devueltos": RC.revisados}),
      f"Tabla 4.6. Búsquedas en GitHub ({RR.repositorio.nunique()} repositorios distintos devueltos)", decimales=0,
      nota="Devueltos: repositorios o archivos descargados, hasta 30 por cadena, ordenados por relevancia.")
tabla(pd.DataFrame({"Repositorio": RV.repositorio, "Archivo": RV.archivo_revisado, "Revisión": RV.tipo_de_revision, "Familia": RV.familia, "¿Churn?": RV.churn,
                    "Qué hace": RV.que_hace, "Diferencia con la propuesta": RV.diferencia}),
      "Tabla 4.7. Repositorios más cercanos", decimales=0)
"""),
md(r"""
Casi todos los resultados son ruido: "mixture of experts" aparece sobre todo en modelos de lenguaje, y *churn* también nombra
la rotación de código. Ningún repositorio de la tabla 4.7 implementa la regla propuesta. El más cercano divide a los clientes
en regímenes con reglas de negocio, entrena un XGBoost por régimen y los combina con una regresión logística; como los
indicadores de régimen entran de forma aditiva, el coeficiente de cada experto es el mismo en todos los regímenes. La
Dirichlet aparece en varios proyectos de churn, pero como generador de pesos globales candidatos y no como prior. Esta
revisión se limita a GitHub y a los primeros 30 resultados de cada cadena; de esos, se leyeron con detalle los de la tabla 4.7.
"""),
]
