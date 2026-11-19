# Predicción de abandono de clientes en Telco Customer Churn

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Book](https://img.shields.io/badge/Jupyter_Book-0.15-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://nataly-cardenas.github.io/Telco_Churn_JBook/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24268653.svg)](https://doi.org/10.5281/zenodo.24268653)
[![Licencia MIT](https://img.shields.io/badge/Licencia-MIT-1B4F72?style=for-the-badge)](LICENSE)

Este repositorio reúne el proyecto de Machine Learning sobre el abandono de clientes (*churn*) de una empresa de
telecomunicaciones, desarrollado en la asignatura Visualización de Datos y Machine Learning de la Universidad del Norte.
Contiene los datos, los notebooks, los resultados guardados de todos los experimentos y el Jupyter Book que los presenta.

- **Libro en línea:** <https://nataly-cardenas.github.io/Telco_Churn_JBook/>
- **Dashboard:** <https://telco-churn-dashboard-hom3.onrender.com/>
- **Archivo permanente:** [doi.org/10.5281/zenodo.24268653](https://doi.org/10.5281/zenodo.24268653)

---

## De qué trata el proyecto

El conjunto público Telco Customer Churn de IBM describe a 7043 clientes con 21 columnas: servicios contratados, datos
de la cuenta, datos demográficos y si el cliente abandonó el servicio en el último mes (26,5 % lo hizo). El objetivo es
identificar a los clientes con mayor riesgo de abandono y los factores asociados, sin generar un exceso de falsas
alertas. El trabajo tiene dos etapas.

**1. Comparación de modelos de referencia.** Tras un análisis exploratorio hecho solo con el conjunto de entrenamiento y
un preprocesamiento sin fuga de información, se evaluaron 112 configuraciones (7 modelos, 4 técnicas de balanceo y 4
métodos de optimización de hiperparámetros) con validación cruzada anidada. Random Forest y XGBoost con SMOTE
alcanzaron el mayor F1 de la clase de abandono (0,638), seguidos de la regresión logística (0,634), sin diferencias
estadísticamente detectables entre ellos. El tipo de contrato, el servicio de internet y la antigüedad son las
variables más influyentes según SHAP.

**2. Modelo propio: RegionalMAPEM.** Es una regla para combinar clasificadores calibrados con pesos que cambian según la
región del espacio de clientes. Los pesos se estiman por máxima posterior con EM y un prior de Dirichlet hacia los pesos
globales, con demostración de convergencia al óptimo global. Se comparó con 11 benchmarks bajo un protocolo
pre-registrado (validación anidada 5 × 5, 223 comparaciones con corrección de Holm y un hold-out evaluado una sola
vez) en Telco y en tres conjuntos adicionales. El resultado es negativo: empata con el mejor benchmark, la logística
con splines (log-loss de 0,41127 frente a 0,41130), supera a 7 de los 11 benchmarks y no cumple la definición de
superioridad. La ablación muestra que en Telco las regiones no aportan frente a los pesos globales.

---

## Contenido del libro

| Capítulo | Archivo | Contenido |
|---|---|---|
| Introducción | `intro.md` | Resumen ejecutivo, problema, objetivos, marco teórico y diccionario de variables |
| Trazabilidad de correcciones | `01_correcciones.md` | Observaciones del entregable anterior, criterios del modelo original y auditoría de fuga de información |
| Análisis exploratorio | `EDA_.ipynb` | Calidad de los datos, partición 80/20, distribuciones, tasas de abandono y asociación entre variables |
| Preprocesamiento | `PREPROCESAMIENTO.ipynb` | Imputación, variables derivadas, codificación, escalado y modelo base de regresión logística |
| Modelado avanzado | `MODELADO_AVANZADO.ipynb` | Las 112 configuraciones, pruebas estadísticas, calibración, multifidelidad, costo computacional e interpretabilidad |
| 4. Literatura y novedad | `04_literatura.ipynb` | Búsquedas en bases institucionales y abiertas, búsqueda por citación, repositorios afines y declaración de novedad |
| 5. Modelo propio | `MODELO_PROPIO.ipynb` | Formulación de RegionalMAPEM, derivación del EM, demostración, algoritmo, complejidad, implementación y pilotos |
| 6. Validación estadística | `06_validacion.ipynb` | Protocolo, equidad del presupuesto, pruebas de hipótesis, ablación y veredicto de superioridad |
| 7. Evaluación ampliada | `07_evaluacion_ampliada.ipynb` | Conjuntos adicionales, sensibilidad, estabilidad, ruido, escalabilidad, calibración, predicción conforme, errores y SHAP |
| 8. Dashboard | `08_dashboard.md` | Arquitectura, capturas, artefactos, ficha del modelo y despliegue en Render |
| 9. Conclusiones | `09_conclusiones.md` | Hallazgos, veredicto sobre novedad, implementación y superioridad, limitaciones y trabajo futuro |
| 10. Reproducibilidad | `10_reproducibilidad.ipynb` | Entorno, semillas, espacios de búsqueda, tabla maestra de experimentos, tiempos y bibliografía |
| Apéndice | `CODIGO_EXPERIMENTO.ipynb` | Código exacto del entrenamiento de las 112 configuraciones |
| Apéndice | `paper.md` | Versión web del paper |
| Apéndice | `README.md` | Esta guía |

---

## Estructura del repositorio

```text
Telco_Churn_JBook/
│
├── intro.md, *.md, *.ipynb      # Capítulos del libro (ver la tabla anterior)
├── _config.yml, _toc.yml        # Configuración e índice del libro
├── references.bib               # Bibliografía
├── style/                       # Estilos del libro
├── Imagenes/                    # Logo, favicon y capturas del dashboard
├── paper/                       # Papers en PDF, fuente LaTeX, figuras y tablas
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Datos originales (IBM)
├── train_index.csv, test_index.csv       # Partición 80/20 (desarrollo y hold-out)
├── X_*_processed.csv, y_*.csv            # Conjuntos procesados del preprocesamiento
│
├── src/original_model.py        # Implementación de RegionalMAPEM (numpy y scipy, interfaz de scikit-learn)
├── src/presentacion.py          # Formato de tablas y figuras de los capítulos de resultados
├── tests/                       # Pruebas unitarias del modelo y de las pruebas estadísticas
├── complementos/                # Calibración isotónica, multifidelidad y optimización computacional
├── resultados_modelado/         # Resultados guardados de las 112 configuraciones
├── experiments/                 # Protocolo, scripts y resultados de la evaluación del modelo propio
│
├── MODEL_CARD.md                # Ficha del modelo RegionalMAPEM
├── DATA_STATEMENT.md            # Declaración de los datos
├── CITATION.cff, LICENSE        # Cita y licencia MIT
├── requirements.txt             # Dependencias del análisis (versiones exactas)
├── requirements-book.txt        # Dependencias para construir el libro
├── verificar_reproduccion.py    # Verifica entorno, datos y artefactos guardados
└── notebooks.sha256             # Huellas de verificación de los notebooks
```

Dentro de `experiments/`:

| Carpeta o archivo | Contenido |
|---|---|
| `protocolo.md` | Protocolo de la evaluación, fijado con el tag de Git `protocolo-v1` antes de los experimentos definitivos, con sus desviaciones |
| `espacios.json`, `familia_holm.json` | Espacios de búsqueda de hiperparámetros y la familia de 223 comparaciones |
| `datos/` | Los tres conjuntos adicionales (Iranian Churn, OpenML 40701 y KDDCup09) con sus huellas SHA-256 |
| `confirmatorio/` | Scripts de la validación anidada, el análisis estadístico, la robustez, el hold-out y la construcción de los capítulos 6, 7 y 10 |
| `resultados/` | Tablas versionadas que leen los capítulos y las tablas selladas antes de abrir el hold-out |
| `literatura/` | Registro de búsquedas, matriz de novedad, búsqueda por citación y revisión de repositorios |
| `pilotos/` | Pilotos exploratorios anteriores al protocolo |

---

## Requisitos y configuración

Se necesita Python 3.12.

1. Clonar el repositorio y entrar en la carpeta:

   ```bash
   git clone https://github.com/Nataly-Cardenas/Telco_Churn_JBook.git
   cd Telco_Churn_JBook
   ```

2. Crear y activar un entorno virtual:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

   Con Anaconda o Miniconda también sirve `conda create --name telco python=3.12 -y` y `conda activate telco`.

3. Instalar las dependencias con sus versiones exactas:

   ```bash
   pip install -r requirements.txt
   ```

4. Verificar el entorno y los resultados guardados:

   ```bash
   python verificar_reproduccion.py
   ```

   El script compara las versiones instaladas con `requirements.txt`, las huellas SHA-256 de los datos con el
   manifiesto del entrenamiento y las huellas de los artefactos de las 1680 unidades del experimento. Con `--rapida`
   omite este último paso.

---

## Cómo reproducir los resultados

Los notebooks del libro no reentrenan modelos: leen los resultados versionados. Para volver a generarlos:

**Primera etapa (112 configuraciones).** Ejecutar en orden `EDA_.ipynb`, `PREPROCESAMIENTO.ipynb` y
`MODELADO_AVANZADO.ipynb`. El código del entrenamiento está en `CODIGO_EXPERIMENTO.ipynb` y el de los experimentos
complementarios en `complementos/` (`python -m complementos.<modulo>`). Reentrenar todo desde cero toma varios días en
un portátil.

**Segunda etapa (modelo propio).**

| Paso | Comando |
|---|---|
| Validación anidada de los 4 conjuntos | `bash experiments/confirmatorio/correr_con_reintentos.sh 10` |
| Combinaciones del dashboard, robustez y cambio por antigüedad | `bash experiments/confirmatorio/correr_etapas_post.sh 10` |
| Análisis confirmatorio y sello previo al hold-out | `python experiments/confirmatorio/analizar.py` y `python experiments/confirmatorio/sellar_pre_holdout.py` |
| Hold-out, análisis final, escalabilidad y SHAP | `bash experiments/confirmatorio/correr_etapas_post.sh 10 final` |
| Tablas del libro | `python experiments/confirmatorio/derivar_tablas_libro.py` |
| Capítulos 6, 7 y 10 | `python experiments/confirmatorio/construir_capitulos.py` |

Con las semillas fijas, una corrida nueva produce los mismos números. El capítulo de reproducibilidad detalla los
tiempos, el hardware y la derivación de las semillas.

**Pruebas unitarias.** `python -m pytest -v tests/` ejecuta las pruebas del modelo propio y de las pruebas
estadísticas; GitHub Actions las corre en cada push.

---

## Construir el libro

El libro se publica con las salidas guardadas en los notebooks: la construcción no los ejecuta ni los modifica.

```bash
pip install -r requirements-book.txt
jupyter-book build .
```

El sitio queda en `_build/html/`; basta con abrir `_build/html/intro.html` en el navegador. Al actualizar la rama
`main`, GitHub Actions construye el libro, verifica las huellas de `notebooks.sha256` y lo publica en GitHub Pages.

---

## Recursos relacionados

- Dashboard interactivo (Plotly Dash): [Dash_Telco_Churn_ML](https://github.com/lauraformore/Dash_Telco_Churn_ML)
- Versión del libro en R (Bookdown): [Telco_Churn_Bookdown](https://github.com/Nataly-Cardenas/Telco_Churn_Bookdown)
- Paper de la comparación de modelos: [`paper/paper.pdf`](paper/paper.pdf)
- Manuscrito del modelo propuesto: [`paper/paper_regionalmapem.pdf`](paper/paper_regionalmapem.pdf)
- Conjunto de datos: [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Cómo citar

Rivera Figueredo, L. S. y Cárdenas Izaquita, N. A. (2026). *Predicción de abandono de clientes en Telco Customer Churn
y modelo RegionalMAPEM* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24268653

Los datos de la cita también están en `CITATION.cff`. El código se distribuye con licencia MIT.

---

## Autoras

- Laura S. Rivera Figueredo (`sriveral@uninorte.edu.co`)
- Natalý A. Cárdenas Izaquita (`nizaquita@uninorte.edu.co`)

Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia.
