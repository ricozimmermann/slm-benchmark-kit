# Metodología Estadística del SLM Benchmark Kit

## 0. Justificación científica: por qué SLMs y por qué un benchmark

El interés por los Small Language Models (SLMs) es coherente con un cambio de enfoque en la IA aplicada: pasar de soluciones basadas exclusivamente en la máxima escala a modelos eficientes, ejecutables en escenarios con restricciones reales de costo, energía, latencia y privacidad.

En la práctica, los SLMs son especialmente relevantes porque:
- reducen los requisitos de memoria y procesamiento, lo que permite su uso en edge/mobile y en entornos con infraestructura limitada;
- permiten procesamiento local/offline, con menor dependencia de la nube y menor exposición de datos sensibles;
- favorecen la democratización tecnológica al ampliar el acceso para instituciones con menor capacidad de cómputo;
- pueden ofrecer un rendimiento competitivo en dominios específicos cuando se ajustan y evalúan correctamente.

Sin embargo, una mayor eficiencia no implica automáticamente un mejor desempeño útil. Por eso, la comparación entre modelos debe hacerse bajo un protocolo experimental controlado, con inferencia estadística explícita y métricas operativas complementarias.

La justificación del benchmark en este proyecto es, por tanto, doble:
1. metodológica: medir diferencias de desempeño controlando la incertidumbre y los factores de confusión;
2. aplicada: identificar el mejor equilibrio entre calidad, estabilidad, latencia y costo de cómputo para escenarios reales.

En síntesis, los SLMs son estratégicos por eficiencia y accesibilidad; un benchmark reproducible es estratégico porque vuelve confiable la decisión técnica.

## 1. Resumen ejecutivo

Este documento formaliza el contenido estadístico utilizado en el proyecto `slm-benchmark-kit` para comparar SLMs con foco en:
- estimación con incertidumbre explícita;
- comparación inferencial entre modelos;
- control de covariables experimentales;
- diagnóstico de calidad del sistema de evaluación automática (judges).

La pipeline actual combina:
- estadística descriptiva por modelo;
- intervalo de confianza bootstrap para la media del score;
- prueba t de Welch para comparación entre dos modelos;
- tamaño del efecto de Cohen (d);
- regresión OLS con errores robustos (cluster por item cuando está disponible; HC3 como fallback);
- métricas de concordancia y salud de los jueces.

## 2. Unidad de análisis y diseño experimental

## 2.1 Unidad observacional

La unidad observacional es el `trial` registrado en el JSONL de resultados, que contiene:
- identificación del modelo (`model`);
- identificación del item (`item_id`), tipo de tarea y dificultad;
- hiperparámetros de generación (`temperature`, `top_p`, `top_k`);
- repetición (`repetition`);
- score agregado (`score_aggregated`) cuando hay al menos un juez válido.

## 2.2 Definiciones operativas

- `valid_response`: en el contexto de la pipeline actual, significa que al menos un juez devolvió un score válido (`judge_valid_count > 0`).
- `score_aggregated`: mediana de los scores válidos de los jueces para el trial.
- `error`: error de generación (por ejemplo, timeout). Cuando está presente, el trial suele quedar sin score agregado.

## 2.3 Estrategias para reducir sesgos

El diseño implementado en el benchmark contempla:
- semilla fija para reproducibilidad;
- randomización del orden de ejecución de los trials;
- repeticiones por combinación de parámetros;
- separación por split de evaluación (`eval_split`) para evitar leakage entre tuning y test;
- evaluación por múltiples jueces con agregación robusta por mediana.

## 3. Variables y métricas reportadas

## 3.1 Rendimiento principal

Para cada modelo, se reporta:
- `n`: total de trials;
- `n_scored`: cantidad de trials con `score_aggregated` numérico;
- media, desvío estándar muestral y mediana de `score_aggregated`;
- intervalo de confianza bootstrap del 95% para la media (`ci95_low`, `ci95_high`).

## 3.2 Confiabilidad operativa

También se reporta:
- `valid_rate` (media de `valid_response`);
- `error_rate` (proporción de errores no vacíos);
- `timeout_rate` (proporción de errores que contienen "timeout");
- `judge_all_failed_rate` (proporción de trials con `judge_valid_count <= 0`);
- `latency_p50_ms` y `latency_p95_ms`.

Estas métricas evitan conclusiones basadas solo en el score medio y hacen visible el costo y la estabilidad.

## 4. Métodos de inferencia estadística

## 4.1 Intervalo de confianza bootstrap para la media

Para cada modelo, el IC del 95% de la media se estima mediante bootstrap percentile:
1. muestrear con reemplazo, `n_boot = 5000`, del vector de scores válidos;
2. calcular la media en cada remuestreo;
3. usar los percentiles 2.5% y 97.5% de la distribución bootstrap.

Configuración actual:
- `seed = 42` en el bootstrap.

Interpretación:
- un IC más estrecho indica una estimación media más precisa;
- la superposición de ICs no sustituye una prueba formal de hipótesis.

## 4.2 Prueba t de Welch (comparación entre dos modelos)

Cuando hay exactamente dos modelos con al menos 2 observaciones válidas por grupo, se aplica la prueba t de Welch (varianzas desiguales):

- Hipótesis nula: las medias poblacionales son iguales.
- Hipótesis alternativa: las medias poblacionales son diferentes.

El informe incluye:
- `t_stat`;
- `p_value`;
- indicador `significant_0_05`.

Nota de alcance:
- la implementación actual ejecuta Welch solo cuando hay 2 modelos; para más de 2 modelos, la prueba se omite con un aviso.

## 4.3 Tamaño del efecto (Cohen's d)

Junto con Welch, se calcula Cohen's d para cuantificar la magnitud del efecto:

- d ~ 0.2: efecto pequeño (regla práctica);
- d ~ 0.5: efecto mediano;
- d ~ 0.8: efecto grande.

Buenas prácticas:
- interpretar `p_value` junto con `d`;
- reportar también la diferencia de medias (`mean_a - mean_b`) y el contexto aplicado.

## 4.4 Regresión OLS con errores robustos

Para controlar la confusión de configuración y la composición de tareas, el proyecto ajusta:

`score_aggregated ~ C(model) + temperature + top_p + top_k + C(task_type)`

Detalles:
- `model` y `task_type` entran como factores categóricos;
- el ajuste solo se realiza si existen las columnas obligatorias y al menos 8 observaciones válidas;
- si `item_id` tiene más de un valor único, se usa covarianza cluster-robust por item;
- en caso contrario (o si falla), se usa HC3.

Interpretación recomendada:
- los coeficientes de `C(model)` representan diferencias condicionales (controladas por covariables);
- los errores robustos reducen la sensibilidad a la heterocedasticidad;
- el cluster por item ayuda a tratar la dependencia intra-item entre repeticiones.

## 5. Calidad del sistema de evaluación automática (judges)

## 5.1 Concordancia par a par

A partir de `judge_scores`, el sistema calcula por par de jueces:
- Spearman;
- Kendall tau;
- Pearson;
- MAE (error absoluto medio);
- `n_overlap` (muestras con score válido en ambos).

Tratamiento de casos límite:
- si `n_overlap < 3`, las correlaciones se reportan como `NaN`;
- si una serie es constante, las correlaciones son `NaN` y el MAE sigue calculándose.

## 5.2 Salud de los jueces

A partir de `judge_rationales`, el sistema estima por juez:
- `valid_rate`;
- `parse_error_rate`;
- `judge_error_rate`;
- `parse_fallback_rate`.

Estas métricas permiten distinguir entre:
- divergencia de criterio (baja concordancia);
- falla técnica de parseo/infraestructura (baja salud operativa).

## 6. Amenazas a la validez y limitaciones

## 6.1 Validez interna

- Dependencia entre trials: las repeticiones del mismo `item_id` no son independientes en sentido estricto (mitigado parcialmente con cluster-robust en OLS).
- Selección por respuestas válidas: los análisis de score usan `score_aggregated` no nulo; las diferencias en la tasa de fallos entre modelos pueden sesgar las comparaciones de medias.

## 6.2 Validez externa

- Los resultados dependen del dataset actual (`slm_tasks_ptbr.jsonl`), del idioma y de la distribución de tareas.
- La generalización a otros dominios requiere replicación con nuevos conjuntos estratificados.

## 6.3 Validez de constructo

- El score agregado depende de la rúbrica de los jueces (heurística/SLM o LLM) y de la robustez del parseo.
- Una alta concordancia no implica validez semántica total; por eso se recomienda una evaluación humana ciega complementaria.

## 7. Reproducibilidad y reporte científico

Para un reporte auditable, se recomienda publicar siempre:
- JSONL bruto;
- informe markdown generado por la pipeline;
- YAML exacto de configuración;
- hash del commit;
- versión del dataset y checksum;
- versión del protocolo (`VERSION` + `CHANGELOG.md`).

Checklist operativo adicional:
1. ejecutar el benchmark con `eval_split: test` para conclusiones finales;
2. registrar fallos (`error_rate`, `timeout_rate`) junto con los scores;
3. reportar efecto + incertidumbre (diferencia de medias, IC, `p_value`, `d`);
4. evitar conclusiones basadas exclusivamente en el umbral de 0.05.

## 8. Directrices de interpretación

Al comparar modelos, priorice este orden:
1. viabilidad operativa (valid_rate, error_rate, timeout_rate, latencia);
2. estimación central e incertidumbre (media + IC bootstrap);
3. evidencia inferencial (Welch + tamaño del efecto);
4. análisis ajustado (OLS robusto) para robustez de la conclusión.

Un modelo con media más alta pero una tasa de fallo elevada puede ser peor en un escenario real que un modelo con media ligeramente menor y mayor estabilidad.

## 9. Referencias metodológicas (ABNT)

- CAMERON, A. Colin; MILLER, Douglas L. A Practitioner's Guide to Cluster-Robust Inference. Journal of Human Resources, v. 50, n. 2, p. 317-372, 2015. DOI: https://doi.org/10.3368/jhr.50.2.317.
- COHEN, Jacob. Statistical Power Analysis for the Behavioral Sciences. 2. ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988. ISBN: 9780805802832.
- EFRON, Bradley; TIBSHIRANI, Robert J. An Introduction to the Bootstrap. New York: Chapman & Hall, 1993. ISBN: 9780412042317.
- MACKINNON, James G.; WHITE, Halbert. Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties. Journal of Econometrics, v. 29, n. 3, p. 305-325, 1985. DOI: https://doi.org/10.1016/0304-4076(85)90158-7.
- WELCH, B. L. The Generalization of Student's Problem when Several Different Population Variances are Involved. Biometrika, v. 34, n. 1-2, p. 28-35, 1947. DOI: https://doi.org/10.2307/2332510.

### 9.1 Nota de rastreabilidad

Las referencias de esta sección fueron verificadas mediante metadatos bibliográficos en fuentes externas:
- artículos: Crossref (título, autores, revista, volumen, número, páginas y DOI);
- libros: catálogos ISBN (Open Library/Google Books) para autores, edición, editorial y año.

Fecha de verificación: 2026-04-15.

### 9.2 Evidencia de uso en repositorios brasileños

Para responder a una posible duda sobre la "vigencia" de los métodos, se realizó una consulta automática adicional en repositorios institucionales brasileños. El objetivo no fue sustituir las referencias fundacionales, sino verificar uso reciente en tesis y disertaciones.

Resumen de resultados observados:
- UFMG ([repositorio.ufmg.br](https://repositorio.ufmg.br/)): búsqueda por bootstrap con 580 resultados; búsqueda por Welch con 394 resultados.
- UFPE ([repositorio.ufpe.br](https://repositorio.ufpe.br/)): búsqueda por bootstrap con 1196 resultados; búsqueda por Welch con 766 resultados.
- UFF ([app.uff.br/riuff](https://app.uff.br/riuff/)): búsqueda por bootstrap con 34 resultados; búsqueda por Welch con 35 resultados.

Ejemplos de elementos devueltos por las consultas:
- UFMG: "GLARMA Model for Temporal Data Analysis: ... a bootstrap proposal for inference on model parameters" (2024).
- UFMG: "Caracterizacao de um modelo de sinucleinopatia ..." (2024), con cita explícita de Welch t test en el resumen extraído.
- UFPE: "Aplicacao de Metodos Bootstrap na Construcao de Intervalos de Confianca para os parametros da Distribuicao Gama" (2022).
- UFF: "Concentracoes sericas de 25-hidroxivitamina D ..." (2025), con uso de prueba t de Student o Welch en el resumen extraído.

Interpretación:
- los métodos clásicos (bootstrap, Welch, tamaño del efecto y errores robustos) siguen ampliamente usados en investigaciones recientes;
- por lo tanto, mantener referencias fundacionales es metodológicamente adecuado;
- complementar con evidencia reciente de aplicación mejora la justificación de vigencia.

Limitaciones de la recolección:
- USP, UNICAMP y UFRGS no tuvieron una extracción estructurada estable mediante la herramienta automática utilizada;
- UNIFESP devolvió una nueva estructura DSpace y requeriría refinar las rutas de búsqueda para extracción por término;
- FGV redirigió a un portal institucional diferente, también con necesidad de búsqueda manual asistida;
- UFABC apunta a un catálogo bibliográfico general (no a un repositorio de tesis en el mismo patrón DSpace), lo que exige una estrategia de consulta distinta.

---

Alcance de esta versión:
- Este documento describe la metodología estadística implementada en el código actual en `src/slm_benchmark/analysis.py` y su integración con la generación de resultados en `scripts/analyze_results.py`.