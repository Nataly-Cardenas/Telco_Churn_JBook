# Guía del Repositorio

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Book](https://img.shields.io/badge/Jupyter_Book-0.15-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://nataly-cardenas.github.io/Telco_Churn_Entregable/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

> Resumen del proyecto: este proyecto predice el abandono de clientes (*churn*) de una empresa de telecomunicaciones a partir del conjunto público Telco Customer Churn de IBM (7,043 clientes, 21 columnas y 26.5 % de abandono). El flujo abarca la limpieza y transformación de los datos (ETL), un análisis exploratorio realizado solo sobre el conjunto de entrenamiento, un preprocesamiento sin fuga de información y un diseño experimental de 112 configuraciones: 7 modelos × 4 técnicas de balanceo × 4 métodos de optimización de hiperparámetros, evaluadas con validación cruzada anidada (5 folds externos × 3 internos).
>
> Resultados: Random Forest y XGBoost con SMOTE alcanzan el mayor F1 de la clase de abandono (0.638), seguidos de la regresión logística (0.634), sin diferencias estadísticamente detectables entre ellos. El balanceo con SMOTE es el factor que más mejora el desempeño, y el tipo de contrato, el servicio de internet y la antigüedad son las variables más influyentes según SHAP.

📖 Libro en línea: <https://nataly-cardenas.github.io/Telco_Churn_Entregable/>

---

## Inicio rápido (requisitos y configuración)

Se necesita Python 3.12. Los pasos siguientes crean un entorno virtual e instalan las librerías con las que se ejecutaron los notebooks.

<details open>
<summary>Configuración del entorno virtual</summary>

<br>

1. Clonar el repositorio y entrar en la carpeta:

   ```bash
   git clone https://github.com/Nataly-Cardenas/Telco_Churn_Entregable.git
   cd Telco_Churn_Entregable
   ```

2. Crear el entorno virtual con Python 3.12:

   ```bash
   python -m venv .venv
   ```

   Con Anaconda o Miniconda también puede usarse `conda create --name telco python=3.12 -y`.

3. Activar el entorno virtual:

   ```bash
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   # Conda
   conda activate telco
   ```

4. Instalar las dependencias del análisis (versiones exactas):

   ```bash
   pip install -r requirements.txt
   ```

5. Verificar el entorno y los resultados guardados:

   ```bash
   python verificar_reproduccion.py
   ```

   El script compara las versiones instaladas con `requirements.txt`, las huellas SHA-256 de los datos de entrada con el manifiesto que se guardó al entrenar los modelos y las huellas de todos los artefactos de las 1,680 unidades (modelos, predicciones, checkpoints). Con `--rapida` omite este último paso.

</details>

---

## Estructura del repositorio

```text
Telco_Churn_Entregable/
│
├── .github/workflows/book.yml   # Publica el libro en GitHub Pages
├── Imagenes/                    # Logo y favicon del libro
├── paper/                       # Paper (PDF) y sus figuras
├── style/custom.css             # Estilos del libro
├── resultados_modelado/         # Artefactos de los experimentos (ver abajo)
├── complementos/                # Calibración isotónica, multifidelidad y optimización computacional
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Datos originales (IBM)
├── train_index.csv / test_index.csv       # Partición 80/20
├── train_test_split.pkl                   # Partición generada en el EDA
├── X_train_processed.csv / X_test_processed.csv  # Variables procesadas
├── y_train.csv / y_test.csv               # Variable objetivo
│
├── EDA_.ipynb                   # Análisis exploratorio (versión publicada)
├── EDA.ipynb                    # Versión anterior del EDA
├── PREPROCESAMIENTO.ipynb       # Preprocesamiento y modelo base
├── MODELADO_AVANZADO.ipynb      # Experimentos, resultados e interpretación
├── CODIGO_EXPERIMENTO.ipynb     # Apéndice: código exacto del entrenamiento de las 112 configuraciones
├── verificar_reproduccion.py    # Verifica entorno, datos y artefactos guardados
│
├── intro.md                     # Introducción y marco teórico
├── paper.md                     # Versión web del paper
├── README.md                    # Esta guía
│
├── _config.yml / _toc.yml       # Configuración e índice del libro
├── requirements.txt             # Dependencias del análisis
├── requirements-book.txt        # Dependencias para construir el libro
└── notebooks.sha256             # Huellas de verificación de los notebooks
```

---

## Flujo de trabajo del proyecto

Los notebooks se ejecutan en este orden; cada uno usa los archivos que genera el anterior.

| Notebook | Contenido | Archivos que genera |
|---|---|---|
| `EDA_.ipynb` | Carga y limpieza de los datos, partición estratificada 80/20 y análisis exploratorio univariado y bivariado del conjunto de entrenamiento (distribuciones, tasas de abandono, V de Cramér, correlaciones) | `train_index.csv`, `test_index.csv`, `train_test_split.pkl` |
| `PREPROCESAMIENTO.ipynb` | Imputación, colapso de categorías redundantes, variable `AvgChargePerMonth`, codificación One-Hot, estandarización y modelo base de regresión logística | `X_train_processed.csv`, `X_test_processed.csv`, `y_train.csv`, `y_test.csv` |
| `MODELADO_AVANZADO.ipynb` | Comparación de las 112 configuraciones con validación cruzada anidada, pruebas de Friedman/Nemenyi y DeLong con tamaños del efecto (δ de Cliff), calibración Platt e isotónica, Successive Halving, optimizaciones computacionales con aceleración medida, evaluación en test e interpretabilidad con SHAP y LIME | `resultados_modelado/` |
| `CODIGO_EXPERIMENTO.ipynb` | Código exacto del entrenamiento de las 112 configuraciones (apéndice, no se ejecuta al construir el libro) | — |

### Resultados guardados

Todos los resultados que usa el libro están versionados en `resultados_modelado/`, así que el notebook de modelado se ejecuta sobre ellos sin volver a entrenar:

| Carpeta | Contenido |
|---|---|
| `oficial/20260920T030103_.../` | El entrenamiento de las 112 configuraciones: `manifest.json` (protocolo, semillas, versiones y huellas de los datos), `splits.json` (particiones), `source_snapshot.json` (código ejecutado) y una carpeta por cada una de las 1,680 unidades con su Pipeline ajustado, el calibrado, predicciones externas, checkpoints de la búsqueda y un `complete.json` con la huella SHA-256 de cada archivo |
| `analisis_final/` | Tablas derivadas: `unidades.csv` (tabla maestra de las 1,680 unidades), `resumen_112.csv`, trazas *anytime*, diversidad genética, pruebas estadísticas, finalistas e interpretabilidad |
| `complementarios/` | Calibración isotónica de las 560 unidades principales, Successive Halving dentro de la validación anidada y las mediciones de optimización computacional |

El código de ese entrenamiento está en `CODIGO_EXPERIMENTO.ipynb`; el de los experimentos complementarios, en `complementos/` (cada módulo se puede ejecutar con `python -m complementos.<modulo>`). Reejecutarlo todo desde cero toma varios días en un portátil, por eso el flujo recomendado es verificar los artefactos y ejecutar el notebook sobre ellos.

---

## Guía de ejecución

1. Activar el entorno virtual (ver *Inicio rápido*).
2. Abrir los notebooks con Jupyter o VS Code:
   - Jupyter: ejecutar `jupyter lab` desde la carpeta del proyecto y abrir los notebooks en el orden de la tabla anterior.
   - VS Code: abrir la carpeta del proyecto, seleccionar el intérprete del entorno virtual (`Ctrl + Shift + P` → Python: Select Interpreter) y ejecutar los notebooks en orden.

---

## Construir el Jupyter Book localmente

El libro se publica con las salidas guardadas en los notebooks: la construcción no los ejecuta ni los modifica.

```bash
pip install -r requirements-book.txt
jupyter-book build .
```

El sitio queda en `_build/html/`; basta con abrir `_build/html/intro.html` en el navegador.

Publicación: al actualizar la rama `main`, GitHub Actions construye el libro y lo publica en GitHub Pages. El flujo verifica con `notebooks.sha256` que los notebooks conserven sus huellas originales.

---

## Recursos relacionados

- 📊 Dashboard interactivo (Plotly Dash): [Dash_Telco_Churn](https://github.com/Nataly-Cardenas/Dash_Telco_Churn)
- 📄 Paper: [`paper/paper.pdf`](paper/paper.pdf)
- 🗂️ Conjunto de datos: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Autoras

- Laura S. Rivera Figueredo — `sriveral@uninorte.edu.co`
- Natalý A. Cárdenas Izaquita — `nizaquita@uninorte.edu.co`

Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia.
