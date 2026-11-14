# Conclusiones y limitaciones

Empezamos con una hipótesis concreta: los ensambles de churn combinan sus modelos con un único vector de pesos, y si la
fiabilidad de cada modelo cambia según el tipo de cliente, una combinación con pesos por región debería aprovecharlo. De
ahí salió RegionalMAPEM: regiones suaves aprendidas con un GMM sobre el espacio de clientes, pesos por región estimados
con MAP-EM y un prior de Dirichlet que empuja esos pesos hacia los globales. Después de una validación anidada
pre-registrada en cuatro conjuntos, con más de 280 horas de cómputo, la respuesta es clara, y no es la que esperábamos
al empezar.

## Hallazgos

1. **El modelo empata con el mejor benchmark.** En Telco, RegionalMAPEM obtiene un log-loss de 0,41127 frente a 0,41130
   de la logística con splines. Ninguna de las cuatro pruebas confirmatorias distingue la diferencia del azar, el δ de
   Cliff es −0,008 y la prueba bayesiana da más de un 99 % de probabilidad de equivalencia práctica (capítulo 6).
2. **Supera con claridad a los modelos individuales clásicos.** Le gana, con al menos tres pruebas después de Holm
   (contando Nemenyi), a kNN, Naive Bayes, el árbol, Random Forest, XGBoost, SVM y CatBoost en Telco, y a la mayoría de los benchmarks en los
   conjuntos adicionales.
3. **Las regiones no aportan; el prior sí protege.** Quitar las regiones (pesos globales) no empeora nada; quitar el prior
   sí. La sensibilidad lo confirma: el log-loss empeora al agregar regiones y vuelve al nivel de los pesos globales cuando
   el prior es fuerte (capítulo 7, figura 7.1). Los pesos aprendidos se reparten igual en todas las regiones, entre la
   logística con splines y HistGB.
4. **Combinar expertos calibrados es lo que funciona.** Las variantes con pesos aprendidos y el stacking logístico ocupan
   los primeros puestos en los cuatro conjuntos, y el stacking es el mejor en los tres adicionales. Con pocos datos (10 %
   del pliegue de ajuste) la ventaja de combinar es clara: HistGB pierde 0,044 de log-loss frente al modelo propio.
5. **El modelo es estable y está bien calibrado.** La semilla mueve su log-loss en 0,0002 y la partición en 0,0007; la
   predicción conforme cumple la cobertura nominal en las dos clases; en el hold-out queda segundo de 17 en log-loss, AUC y
   ECE, detrás de su variante sin prior.

## Veredicto sobre R1, R2 y R3

| Requisito | Veredicto | Sustento |
|---|---|---|
| **R1. Novedad** | Se cumple, con alcance acotado | La integración de clasificadores por zonas del espacio de clientes ya existe en churn (dos trabajos de 2010 y 2012 con subáreas de k-means y pesos optimizados por metaheurísticas). Lo nuevo es la regla de combinación: pertenencias suaves, expertos fijos que no se reentrenan por región, MAP-EM con objetivo cóncavo y garantía de óptimo global, y un prior de Dirichlet hacia los pesos globales. Se diferencia de la mezcla de expertos con compuerta EM, de la selección dinámica, del stacking con pesos locales y de los ensambles ponderados (capítulos 4 y 5) |
| **R2. Implementación propia** | Se cumple | PCA, GMM y MAP-EM programados en numpy y scipy, con interfaz de scikit-learn, demostración de concavidad y ascenso monótono, y 12 pruebas unitarias del modelo, que pasan todas (capítulo 5) |
| **R3. Superioridad** | No se cumple | De las cuatro condiciones solo se cumple la primera, por 0,00003; ninguna prueba rechaza frente al mejor benchmark, el efecto es despreciable y el modelo es mejor en media en solo uno de los tres conjuntos adicionales (tabla 6.8) |

Con R1 y R2 cumplidos y R3 no alcanzado, el trabajo queda en el tope de 4,0. Es un resultado negativo honesto y
documentado: el protocolo estaba congelado antes de ver los resultados, el hold-out se abrió una sola vez después de
sellar el análisis, y ninguna decisión posterior al tag cambió qué se calculó.

## Por qué no ganó

La causa probable no es falta de potencia. Los 25 pliegues detectan diferencias de 0,0004, como la del prior, y la prueba
bayesiana no deja la conclusión en suspenso: favorece la equivalencia. Tampoco es falta de presupuesto, porque todos los
modelos tuvieron las mismas 30 evaluaciones y ninguna falló.

La explicación está en los datos. En Telco, la fiabilidad relativa de los expertos casi no cambia entre tipos de cliente,
así que no hay heterogeneidad que una combinación por regiones pueda capturar. Las diferencias por región que aparecen en
el análisis de errores van en ambos sentidos y se concentran en regiones de 11 a 30 clientes, que es el patrón del
ruido. Con un único vector de pesos, el modelo se reduce a un buen ensamble ponderado. La logística con splines, por su
parte, ya captura casi toda la señal disponible en 19 variables de contrato y servicio. Incluso en Iranian, el único conjunto donde
el modelo gana en media al mejor benchmark, el stacking logístico rinde todavía mejor.

## Limitaciones

- **Un solo conjunto principal.** Las conclusiones confirmatorias son sobre Telco; los tres adicionales dan consistencia,
  pero cuatro conjuntos no alcanzan para que el Wilcoxon por pares entre conjuntos pueda rechazar (su p mínimo es 0,125).
- **KDDCup09 con 2 repeticiones.** Por costo de cómputo, decidido antes de ver resultados. Sus pruebas tienen 10
  pliegues en lugar de 25.
- **El espacio de las regiones es fijo.** Las regiones se construyen sobre dos componentes principales de las variables
  continuas. Otra elección (variables categóricas, más componentes, regiones supervisadas) podría encontrar
  heterogeneidad que esta no ve, pero probarla ahora sería ajustar el modelo a los resultados.
- **El costo en producción es el de once modelos.** La parte propia es barata, pero el modelo necesita a todos los
  expertos para predecir: unos 23 MB serializados y el tiempo de inferencia de Random Forest y SVM.
- **Robustez sobre una repetición.** Los estudios exploratorios usan los 5 pliegues de la repetición 0, sin volver a
  optimizar.
- **Revisión de literatura.** La búsqueda por citación se hizo en OpenAlex y los dos trabajos de 2010 y 2012 se analizaron
  a partir de sus resúmenes, lo que limita el alcance de la declaración de novedad (capítulo 4).

## Riesgos de uso

Un modelo de churn decide a quién se le ofrece retención, así que sus errores tienen consecuencias asimétricas. Las
probabilidades están bien calibradas en promedio, pero un tercio de los clientes recibe un conjunto conforme con las dos
clases: para ellos el modelo no tiene certeza y conviene una revisión humana antes de actuar. El conjunto de datos es una
muestra sintética de IBM de un solo operador; no debe usarse para decidir sobre clientes de otro mercado sin volver a
validarlo. Las variables no incluyen atributos protegidos más allá de `gender` y `SeniorCitizen`, que tienen un peso
SHAP casi nulo, pero no hicimos un análisis formal de equidad.

## Trabajo futuro

- Probar el modelo donde la heterogeneidad es plausible por diseño: conjuntos con segmentos de negocio claramente
  distintos (residencial frente a empresas, prepago frente a pospago) o con más variables de uso, como Iranian.
- Construir las regiones con información supervisada, por ejemplo con el desacuerdo entre expertos, en lugar de solo con
  la geometría de las variables.
- Reducir el número de expertos: los pesos muestran que dos o tres concentran casi todo, y podarlos abarataría el
  modelo sin perder calidad.
- Comparar con mezclas de expertos con compuerta entrenada de punta a punta, que pueden adaptar a la vez las regiones y
  los expertos.

## Declaración de autoría y uso de herramientas de inteligencia artificial

Laura S. Rivera Figueredo y Natalý A. Cárdenas Izaquita somos las autoras de este proyecto. Decidimos el problema,
la hipótesis, el modelo, el protocolo y las conclusiones, y respondemos por todo su contenido.

Durante el proyecto usamos Claude (Anthropic), a través de Claude Code, como asistente de programación y redacción. Lo
usamos para escribir y revisar código de los scripts de experimentación y análisis estadístico, la clase del modelo y
sus pruebas, para auditar tiempos de cómputo y la coherencia entre el protocolo y lo ejecutado, y para redactar
borradores de varios capítulos a partir de los resultados. Todas sus propuestas pasaron por nuestra revisión. Los
resultados numéricos salen de la ejecución del código del repositorio, no de la herramienta, y ninguna cifra del libro
se escribió sin verificarla contra las tablas versionadas. La herramienta no tuvo acceso al hold-out antes de su
evaluación única, que se hizo después de sellar el análisis confirmatorio.
Además, usamos Claude (Anthropic), como herramienta de apoyo científico, para una validación breve del manuscrito y de la investigación bibliográfica: contrastar el texto con las tablas versionadas y localizar y verificar referencias. Los hallazgos se comprobaron contra las fuentes antes de incorporarlos. Este trabajo no recibió ningún tipo de financiación.

La declaración firmada por el equipo está al final del capítulo 10.
