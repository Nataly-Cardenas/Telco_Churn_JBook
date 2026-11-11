# Piloto exploratorio: cota del oráculo (2026-11-07)

**Carácter:** análisis exploratorio, previo al protocolo pre-registrado. No sustenta R3 y debe contarse
como iteración de diseño (Sección 5.3 de la guía). No usa el hold-out de prueba.

**Script:** `piloto_oraculo.py` · **Salidas:** `resultados_oraculo/` · **Log:** `log_oraculo.txt`

## Diseño

- Conjunto de desarrollo del entregable 3 (5634 clientes, 26,5 % de churn).
- 5 repeticiones × 5 pliegues estratificados (25 pliegues externos). La repetición 0 reproduce los
  pliegues externos del entregable 3 (`seed_for('outer', 0)`).
- 9 expertos, sin SMOTE y todos calibrados con Platt dentro del pliegue de ajuste. Tienen
  hiperparámetros fijos y no optimizados: el piloto mide techos relativos, no el desempeño final.
- Combinadores honestos: pesos aprendidos con predicciones fuera de pliegue (3 pliegues internos).
- Oráculos: pesos ajustados sobre el mismo pliegue de evaluación. Son una cota superior optimista
  porque incluyen el sobreajuste a ese pliegue.
- Regiones: GMM sobre `tenure` y `MonthlyCharges`, estandarizadas con la media y la desviación del
  pliegue de ajuste.

## Resultados (log-loss medio en los 25 pliegues; ganancia frente al mejor experto, la logística con splines, 0,4128)

| Método | Ganancia | Pliegues mejores | δ de Cliff | Wilcoxon p | Nadeau-Bengio p | Permutación p |
|---|---|---|---|---|---|---|
| Stacking logístico (honesto) | 0,0008 | 18/25 | 0,04 | 0,059 | 0,42 | 0,20 |
| EM con pesos globales (honesto) | 0,0007 | 18/25 | 0,04 | 0,019 | 0,36 | 0,27 |
| MAP-EM por regiones, K=4, c=10 (honesto) | 0,0007 | 17/25 | 0,04 | 0,030 | 0,38 | 0,19 |
| Oráculo, pesos globales | 0,0021 | 24/25 | 0,16 | <0,001 | 0,008 | 0,015 |
| Oráculo por regiones, K=2 | 0,0037 | 25/25 | 0,24 | <0,001 | <0,001 | <0,001 |
| Oráculo por regiones, K=4 | 0,0049 | 25/25 | 0,30 | <0,001 | <0,001 | <0,001 |
| Oráculo por regiones, K=8 | 0,0076 | 25/25 | 0,45 | <0,001 | <0,001 | <0,001 |

Otros datos:

- Ganancia necesaria para que |δ| ≥ 0,147 con la dispersión observada entre pliegues: unos 0,0024.
- El ascenso de la log-posterior de MAP-EM fue monótono en todos los pliegues y en todas las
  configuraciones (K, c).
- Las predicciones de los expertos están muy correlacionadas: la correlación media entre sus logits
  es 0,88.
- Con c = 0 (sin prior), los pesos por región sobreajustan y empeoran a medida que crece K. Con c = 10
  y c = 100, la regla se encoge hacia los pesos globales y queda igual que ellos.

## Lectura

1. La ganancia honesta de cualquier combinación (0,0007–0,0008) es cerca de un tercio de la necesaria
   para un efecto no despreciable. Ni Nadeau-Bengio ni la permutación la detectan.
2. Que el oráculo por regiones mejore por sí solo no prueba que exista un margen aprovechable. La ganancia
   crece casi linealmente con el número de pesos libres (K × 8), lo que indica sobreajuste al pliegue
   de evaluación. En los pesos globales, el oráculo gana 0,0021 y la versión honesta 0,0007: unos
   0,0014 de optimismo con solo 8 pesos libres.
3. La heterogeneidad regional que se ve dentro de la muestra no se generaliza: MAP-EM por regiones no
   mejora a los pesos globales fuera de la muestra. El prior de Dirichlet cumple su función, porque
   evita el deterioro que sufre la versión sin prior.
4. Conclusión para el protocolo: con esta familia de modelos y este dataset, R3 no es alcanzable sin
   forzar el resultado. Se recomienda pre-registrar la hipótesis de superioridad esperando un resultado
   negativo y usar este piloto como evidencia de la causa probable: un techo de información del dataset,
   con expertos casi redundantes.

## Limitaciones del piloto

- Los expertos no están optimizados. Con benchmarks optimizados, el margen de cualquier combinación
  suele ser igual o menor.
- No se probaron CatBoost ni LightGBM porque no están instalados en el entorno `ml_project`.
- Las regiones se definieron con dos variables numéricas. Otras definiciones (por ejemplo, el tipo de
  contrato) quedan como iteraciones de diseño posibles, y deben contarse como tales si se exploran.

---

# Piloto exploratorio 2: híbrido GLM + boosting residual (2026-11-07)

**Script:** `piloto_hibrido.py` · **Salidas:** `resultados_oraculo/hibrido_*.csv`. Usa los mismos 25 pliegues y semillas
que el piloto 1 y no usa el hold-out. Es la segunda iteración de diseño exploratoria.

Híbrido: logística con splines como margen fijo (offset) y árboles poco profundos que se ajustan solo a lo que
la parte aditiva no explica. Se reportan las 5 configuraciones probadas, no solo la mejor.

| Método | Log-loss | Ganancia vs. logística con splines (Platt) | Pliegues mejores | Nadeau-Bengio p |
|---|---|---|---|---|
| Híbrido (5 configuraciones) | 0,4106–0,4108 | 0,0019–0,0021 | 21–24/25 | 0,002–0,039 |
| Logística con splines, 20 nudos por cuantiles (control) | 0,4113 | 0,0014 | 18/25 | 0,38 |
| Logística con splines, 10 nudos por cuantiles (control) | 0,4116 | 0,0011 | 18/25 | 0,23 |
| Logística con splines sin Platt, C entre 0,2 y 2 (control) | 0,4126–0,4129 | ≈0 | — | >0,4 |

## Lectura

- La ganancia del híbrido es estable entre configuraciones y no se debe ni a la calibración ni a la regularización.
- Unos dos tercios de esa ganancia los consigue un benchmark simple: la misma logística con más nudos por
  cuantiles. Frente a ese benchmark, el híbrido solo gana unos 0,0007, el mismo orden que los combinadores
  del piloto 1 y lejos de los 0,0024 que se necesitan para un efecto no despreciable.
- Que los árboles de profundidad 1 (sin interacciones) ganen lo mismo que los de profundidad 3 indica que
  lo que falta es forma no lineal aditiva, no interacciones. Eso es justo lo que un GAM bien ajustado ya
  modela.
- Consecuencia para el protocolo: una logística con splines de nudos optimizados debe estar entre los
  benchmarks. Sin ella, cualquier modelo propio parecería mejor de lo que es.

# Piloto 8: ¿las regiones capturan heterogeneidad real?

Script: `piloto_heterogeneidad.py`. Usa la caché de `piloto_kaggle/` (25 pliegues de desarrollo, sin
hold-out). Se ejecuta desde la raíz del repositorio:

    python experiments/pilotos/piloto_heterogeneidad.py experiments/pilotos/piloto_kaggle/cache

Que el mejor experto cambie entre regiones no prueba nada si los expertos están casi empatados. En cada
pliegue se calcula T, la ganancia de log-verosimilitud por cliente de los pesos por región (K = 4, c = 0)
frente a los pesos globales, y se compara con 200 permutaciones de las pertenencias a regiones entre
clientes (H0: la fiabilidad relativa de los expertos no depende de la región).

| Medida | Valor |
|---|---|
| T medio, regiones reales | 0,00134 |
| T medio, regiones permutadas | 0,00069 |
| Pliegues con p < 0,05 | 15 de 25 |
| p mediano | 0,040 |
| Pliegues en que cambia el mejor experto (real / bajo H0) | 24 de 25 / 76 % |

## Lectura

- Hay heterogeneidad real, pero débil: unos 0,0006 de log-verosimilitud por cliente por encima del azar,
  medidos dentro de la muestra. Fuera de ella, la ablación del piloto 6b da 0,0001.
- El conteo de cambios del mejor experto no distingue la señal del azar y deja de usarse como evidencia.
- Los pliegues comparten clientes, así que sus p no se combinan; el resultado es descriptivo.

Resultados en `resultados_heterogeneidad/` (`permutacion_por_pliegue.csv`, `resumen.json`).
