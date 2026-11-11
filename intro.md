# Modelo de Clasificación Binaria para la Predicción de Abandono de Clientes a partir de Datos de Suscripción: Dataset Telco Customer Churn

*Análisis de Retención de Clientes y Riesgo de Abandono (Churn)*

Laura S. Rivera Figueredo y Natalý A. Cárdenas Izaquita  
Departamento de Matemáticas, Física y Ciencia de Datos, Universidad del Norte, Barranquilla, Colombia  
`sriveral@uninorte.edu.co` · `nizaquita@uninorte.edu.co`

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Asignatura
:class-card: meta-card
Visualización de Datos y Machine Learning
:::

:::{grid-item-card} Conjunto de datos
:class-card: meta-card
Telco Customer Churn (IBM) · 7,043 clientes · 26.5 % de abandono
:::

:::{grid-item-card} Metodología
:class-card: meta-card
EDA · Modelado avanzado · Modelo original · Validación estadística
:::

::::

## Resumen ejecutivo

Este libro predice qué clientes de una empresa de telecomunicaciones van a abandonar el servicio y, en su última etapa,
pone a prueba un modelo propio. La hipótesis era que los ensambles de churn pierden información al combinar sus modelos
con un único vector de pesos: si la fiabilidad de cada modelo cambia según el tipo de cliente, convendría darle pesos
distintos a cada región del espacio de clientes. Para probarlo diseñamos RegionalMAPEM y lo comparamos con 11 benchmarks
bajo un protocolo congelado antes de correr los experimentos definitivos, en Telco y en tres conjuntos adicionales.

El resultado es negativo y claro. RegionalMAPEM empata con el mejor benchmark, la logística con splines (log-loss de
0,41127 frente a 0,41130), le gana a 7 de los 11 benchmarks y no cumple la definición de superioridad. La ablación y la
sensibilidad muestran por qué: en Telco, las regiones no aportan y el prior de Dirichlet hace bien su trabajo al llevar
los pesos de cada región hacia los globales.

**Contribuciones.**

1. RegionalMAPEM, una regla de combinación de clasificadores calibrados con pesos por región, estimados por MAP-EM con un
   prior de Dirichlet hacia los pesos globales, con demostración de concavidad y convergencia al óptimo global
   (capítulo 5).
2. Una implementación propia en numpy y scipy, compatible con scikit-learn y con 12 pruebas unitarias (capítulo 5).
3. Una evaluación pre-registrada con validación anidada 5 × 5, 223 comparaciones con corrección de Holm y un hold-out
   abierto una sola vez después de sellar el análisis (capítulo 6).
4. Una evaluación ampliada en cuatro conjuntos de churn, con estudios de sensibilidad, estabilidad, ruido, tamaño de
   muestra, cambio de distribución, calibración, predicción conforme, análisis de errores e interpretabilidad (capítulo 7).
5. Un análisis honesto de por qué el modelo no supera al mejor benchmark (capítulos 7 y 9).

## Introducción
La pérdida de clientes (*churn*) es uno de los principales riesgos financieros de las empresas de telecomunicaciones: cada cancelación supone la pérdida de ingresos recurrentes, y retener a un cliente existente suele ser considerablemente más económico que adquirir uno nuevo. Anticipar qué clientes tienen mayor probabilidad de abandonar el servicio permite dirigir ofertas, mejoras de servicio y acciones de retención de forma selectiva, en lugar de aplicarlas a toda la base de clientes.

El problema presenta dos dificultades. Los clientes que abandonan son una minoría (alrededor de uno de cada cuatro), por lo que un modelo puede acertar en la mayoría de los casos sin detectar a casi ninguno de ellos; y las variables disponibles describen el estado del cliente en un momento dado, lo que obliga a interpretar los hallazgos como asociaciones y no como causas del abandono.

Este proyecto aborda la predicción del abandono con el conjunto público Telco Customer Churn de IBM, que combina los servicios contratados, las características de la cuenta y los datos demográficos de cada cliente. A partir de un análisis exploratorio que identifica los perfiles con mayor tasa de abandono, se construye un modelo base de regresión logística y se compara con alternativas más complejas bajo un mismo protocolo de validación, priorizando la detección de los clientes en riesgo sin generar un volumen excesivo de falsas alertas. Los resultados se ponen a disposición en este Jupyter Book, en un paper y en un tablero interactivo.

## Contexto del problema
El *customer churn* es la cancelación o abandono de un servicio. Según los [datos de Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), cada fila del conjunto representa un cliente y la columna `Churn` indica si abandonó el servicio durante el último mes. El conjunto de datos incluye información sobre cuatro grupos de variables.

- El abandono, que identifica a los clientes que se dieron de baja en el último mes (columna `Churn`).
- Los servicios contratados, que comprenden teléfono, líneas múltiples, internet, seguridad en línea, copia de seguridad en línea, protección de dispositivos, soporte técnico y transmisión de TV y películas.
- La información de la cuenta, que reúne la antigüedad como cliente, el tipo de contrato, el método de pago, la facturación electrónica, los cargos mensuales y los cargos totales.
- La información demográfica, que recoge el género, el rango de edad y si tienen pareja o personas a su cargo.

## Objetivo general
Identificar a los clientes de una empresa de telecomunicaciones con mayor riesgo de abandonar el servicio y los factores asociados a ese abandono, construyendo y comparando de forma reproducible modelos de clasificación que detecten la mayor cantidad posible de clientes en riesgo sin generar un exceso de falsas alertas.

### Objetivos específicos
1. Describir el problema y los datos, caracterizando el conjunto Telco Customer Churn, su calidad (valores faltantes, atípicos y tipos de variables) y el nivel de desbalance de la variable objetivo.
2. Reservar un conjunto de prueba, separando el 20 % de los clientes mediante una partición estratificada antes del análisis, de modo que ninguna decisión se tome con los datos destinados a la evaluación final.
3. Identificar los perfiles de abandono, analizando la distribución de cada variable y su asociación con el abandono, para determinar qué segmentos de clientes (tipo de contrato, servicios, antigüedad, método de pago) concentran las mayores tasas de cancelación.
4. Preparar los datos sin fuga de información, definiendo el tratamiento de faltantes, la codificación de variables categóricas, el escalado y las variables derivadas, ajustándolos únicamente con los datos de entrenamiento de cada partición.
5. Establecer un modelo de referencia, entrenando una regresión logística como modelo base y cuantificando sus aciertos y errores, en especial los clientes que abandonan y no son detectados.
6. Comparar alternativas de forma controlada, evaluando 112 configuraciones (7 modelos × 4 técnicas de balanceo × 4 métodos de optimización de hiperparámetros) con validación cruzada anidada y el F1 de la clase de abandono como métrica principal, contrastando estadísticamente sus diferencias, la calibración de probabilidades y el costo computacional.
7. Explicar las predicciones, identificando con SHAP y LIME las variables que más influyen en el riesgo de abandono, sin atribuirles un efecto causal.
8. Poner los resultados a disposición, presentando el análisis en un Jupyter Book, un paper y un tablero interactivo que permitan consultar los hallazgos sin necesidad de ejecutar el código.
9. Proponer un modelo original y evaluarlo con rigor, formulando una regla de combinación nueva, demostrando su propiedad teórica, implementándola desde cero y comparándola con los benchmarks bajo un protocolo pre-registrado, con el mismo presupuesto y en cuatro conjuntos de datos.

## Marco teórico

### Abandono de clientes y clasificación binaria
El abandono de clientes (*churn*) es la cancelación o no renovación de un servicio durante un periodo determinado. Su predicción se plantea como una clasificación binaria supervisada: a partir de las características $x_i$ de cada cliente se estima la probabilidad $P(y_i = 1 \mid x_i)$ de que abandone. El análisis exploratorio previo (Tukey, 1977) mide la asociación de cada variable con el abandono mediante la correlación punto-biserial, la V de Cramér (Cramér, 1946) y el coeficiente de Spearman, con corrección de Holm para contrastes múltiples (Holm, 1979).

### Desbalance de clases y fuga de información
Cuando la clase de interés es minoritaria, los modelos tienden a favorecer a la mayoritaria. SMOTE genera ejemplos sintéticos de la clase minoritaria interpolando entre vecinos cercanos (Chawla et al., 2002) y ADASYN los concentra en las regiones más difíciles (He et al., 2008). Estas técnicas, junto con la imputación, la codificación y el escalado, deben ajustarse solo con los datos de entrenamiento; de lo contrario, información de prueba se filtra al modelo (*data leakage*) y las métricas resultan optimistas.

### Modelos de clasificación
Se usa la regresión logística como modelo base por su interpretabilidad (Hosmer et al., 2013) y se compara con KNN (Cover y Hart, 1967), Naive Bayes (Domingos y Pazzani, 1997), árboles de decisión (Breiman et al., 1984), Random Forest (Breiman, 2001), XGBoost (Chen y Guestrin, 2016) y SVM (Cortes y Vapnik, 1995). Sus hiperparámetros se eligen con Grid Search, Random Search (Bergstra y Bengio, 2012), TPE (Bergstra et al., 2011) o algoritmos genéticos, dentro de una validación cruzada anidada: un bucle interno selecciona los hiperparámetros y uno externo mide el desempeño con datos que no intervinieron en esa selección, evitando estimaciones optimistas (Varma y Simon, 2006).

### Métricas e interpretabilidad
Tomando el abandono como clase positiva:

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2\,TP}{2\,TP + FP + FN}.
$$

El F1 equilibra la detección de clientes que abandonan (Recall) y la proporción de alertas correctas (Precision), por lo que es más informativo que la exactitud con clases desbalanceadas; el AUC-ROC mide la capacidad de ordenar a los clientes por riesgo. Las diferencias entre modelos se contrastan con las pruebas de Friedman y Nemenyi (Demšar, 2006) y de DeLong (DeLong et al., 1988). Por último, SHAP (Lundberg y Lee, 2017) y LIME (Ribeiro et al., 2016) explican qué variables influyen en las predicciones, sin implicar relaciones causales.

## Descripción de las variables del proyecto
Diccionario resumido de los metadatos de Kaggle:

| Variable | Tipo conceptual | Significado | Papel |
|---|---|---|---|
| customerID | Identificador | Código del cliente | Integridad; no predictor |
| gender | Categórica | Género registrado (Masculino/Femenino) | Candidato |
| SeniorCitizen | Binaria | Adulto mayor (1/0) | Candidato |
| Partner | Binaria | Tiene pareja (1/0) | Candidato |
| Dependents | Binaria | Tiene dependientes | Candidato |
| tenure | Numérica discreta | Antigüedad en meses | Candidato |
| PhoneService | Categórica | Servicio telefónico | Candidato |
| MultipleLines | Categórica | Líneas múltiples | Candidato |
| InternetService | Categórica | Servicio de internet | Candidato |
| OnlineSecurity | Categórica | Seguridad en línea | Candidato |
| OnlineBackup | Categórica | Respaldo en línea | Candidato |
| DeviceProtection | Categórica | Protección de dispositivos | Candidato |
| TechSupport | Categórica | Soporte técnico | Candidato |
| StreamingTV | Categórica | Televisión por streaming | Candidato |
| StreamingMovies | Categórica | Películas por streaming | Candidato |
| Contract | Categórica | Modalidad contractual | Candidato |
| PaperlessBilling | Categórica | Facturación sin papel | Candidato |
| PaymentMethod | Categórica | Método de pago | Candidato |
| MonthlyCharges | Numérica | Cargo mensual | Candidato |
| TotalCharges | Numérica; puede venir como texto | Cargos totales | Candidato |
| Churn | Binaria | Abandono durante el último mes | Objetivo Yes/No |


El identificador no representa una característica explicativa. Los demás campos son candidatos.

## Referencias

- Bergstra, J., Bardenet, R., Bengio, Y. y Kégl, B. (2011). Algorithms for hyper-parameter optimization. En *Advances in Neural Information Processing Systems*, 24.
- Bergstra, J. y Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L., Friedman, J. H., Olshen, R. A. y Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
- Chawla, N. V., Bowyer, K. W., Hall, L. O. y Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).
- Cortes, C. y Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Cover, T. y Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
- DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics*, 44(3), 837–845.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
- Domingos, P. y Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. *Machine Learning*, 29, 103–130.
- He, H., Bai, Y., Garcia, E. A. y Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. En *2008 IEEE International Joint Conference on Neural Networks* (pp. 1322–1328).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Hosmer, D. W., Lemeshow, S. y Sturdivant, R. X. (2013). *Applied Logistic Regression* (3.ª ed.). Wiley.
- Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. En *Advances in Neural Information Processing Systems*, 30 (pp. 4765–4774).
- Ribeiro, M. T., Singh, S. y Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135–1144).
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Varma, S. y Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.


## Estructura del libro

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} 1. Trazabilidad de correcciones
:class-card: step-card
Qué se corrigió del entregable anterior, dónde está en el libro y con qué se comprueba, más la auditoría de fuga de información.
:::

:::{grid-item-card} 2. Análisis exploratorio
:class-card: step-card
Calidad de los datos, distribuciones, tasas de abandono por segmento y asociación entre variables.
:::

:::{grid-item-card} 2. Preprocesamiento
:class-card: step-card
Imputación, codificación, escalado y modelo base, ajustados solo con los datos de entrenamiento.
:::

:::{grid-item-card} 3. Modelado avanzado
:class-card: step-card
Las 112 configuraciones del entregable anterior con validación cruzada anidada, calibración y comparación estadística.
:::

:::{grid-item-card} 4. Literatura y novedad
:class-card: step-card
Búsqueda en bases institucionales y abiertas, búsqueda por citación, tabla de diferenciación y declaración de novedad.
:::

:::{grid-item-card} 5. Modelo original
:class-card: step-card
Motivación, formulación, derivación del EM, demostración, pseudocódigo, complejidad, implementación y pruebas.
:::

:::{grid-item-card} 6. Validación estadística
:class-card: step-card
Protocolo pre-registrado, equidad, pruebas de hipótesis con Holm, ablación y veredicto sobre R3.
:::

:::{grid-item-card} 7. Evaluación ampliada
:class-card: step-card
Conjuntos adicionales, sensibilidad, estabilidad, ruido, escalabilidad, cambio de distribución, calibración, incertidumbre, errores y SHAP.
:::

:::{grid-item-card} 9. Conclusiones
:class-card: step-card
Hallazgos, veredicto sobre R1, R2 y R3, limitaciones, riesgos, trabajo futuro y declaración de uso de IA.
:::

:::{grid-item-card} 10. Reproducibilidad
:class-card: step-card
Entorno, semillas, espacios de búsqueda, tabla maestra de experimentos, tiempos y referencias.
:::

:::{grid-item-card} Apéndices
:class-card: step-card
Código del experimento del entregable anterior, paper y guía del repositorio.
:::

::::
