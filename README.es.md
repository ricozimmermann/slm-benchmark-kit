# slm-benchmark-kit

Framework reutilizable para benchmark de SLMs (Small Language Models).

Objetivo:
- comparar modelos y parametros con reproducibilidad;
- soportar multiples proyectos con el mismo protocolo;
- producir analisis estadistico real (Welch t-test con tamano de efecto, OLS robusto por item, IC bootstrap).

## 1. Estrategia metodologica

Este proyecto ya incluye mejoras esenciales de benchmark:
- semilla fija para reproducibilidad;
- aleatorizacion del orden de trials;
- repeticiones por combinacion de parametros;
- split de evaluacion explicito (`eval_split`) para separar tuning y test;
- multi-judge con agregacion por mediana, usando dos jueces LLM con nombre en el benchmark principal;
- analisis estadistico con scipy/statsmodels (sin atajos heurisiticos de p-value).

## 2. Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 2.1 Onboarding rapido (orden recomendado)

Si es tu primera vez en este proyecto, sigue este orden:

1. Ejecuta un smoke test local para validar entorno y Ollama:

```bash
python scripts/run_benchmark.py --config configs/benchmark_local_smoke.yaml --check-local
```

2. Ejecuta el benchmark completo (o tu config personalizada):

```bash
python scripts/run_benchmark.py --config configs/benchmark_ollama.yaml
```

3. Genera el reporte estadistico:

```bash
python scripts/analyze_results.py --input results/raw_benchmark.jsonl --output results/report.md
```

4. (Opcional) Prepara evaluacion humana ciega:

```bash
python scripts/prepare_human_eval.py \
	--input results/raw_benchmark.jsonl \
	--assignment results/human_assignment.csv \
	--key results/human_key_private.csv \
	--evaluators eval01 eval02 eval03 \
	--sample-size 120 \
	--overlap-rate 0.25
```

5. (Opcional) Genera concordancia entre evaluadores:

```bash
python scripts/agreement_report.py \
	--input results/human_assignment_scored.csv \
	--key results/human_key_private.csv \
	--output results/human_agreement.md
```

## 3. Ejecutar benchmark

Comando principal (ya detallado en onboarding 2.1):
- `python scripts/run_benchmark.py --config configs/benchmark_ollama.yaml`

### 3.1 Prueba local rapida (SLM en Ollama)

Usa la configuracion smoke para validar el pipeline localmente con pocas muestras y comparar dos modelos a escala reducida:

```bash
python scripts/run_benchmark.py --config configs/benchmark_local_smoke.yaml --check-local
```

Esto hace:
- verificacion de conectividad con Ollama local;
- advertencia si el modelo configurado no esta descargado localmente;
- ejecucion con `trial_limit` pequeno para validacion rapida.

Salida cruda (JSONL):
- `results/raw_benchmark_local_smoke.jsonl`

## 4. Generar reporte

Comando principal (ya detallado en onboarding 2.1):
- `python scripts/analyze_results.py --input results/raw_benchmark.jsonl --output results/report.md`

## 5. Arquitectura del proyecto (didactica)

### 5.1 Mapa visual del flujo

```text
datasets/*.jsonl
	 -> configs/*.yaml
	 -> scripts/run_benchmark.py
	 -> src/slm_benchmark/runner.py
	 -> results/raw_benchmark*.jsonl
	 -> scripts/analyze_results.py
	 -> src/slm_benchmark/analysis.py
	 -> results/report.md

Opcional (evaluacion humana):
results/raw_benchmark*.jsonl
	 -> scripts/prepare_human_eval.py
	 -> results/human_assignment.csv + results/human_key_private.csv
	 -> (evaluadores completan scores)
	 -> scripts/agreement_report.py
	 -> results/human_agreement.md
```

### 5.2 Carpeta por carpeta

- `configs/`
	- Rol: definir el experimento sin cambiar codigo.
	- `benchmark_local_smoke.yaml`: validacion rapida (pocas combinaciones, `trial_limit` bajo).
	- `benchmark_ollama.yaml`: benchmark principal (mas modelos/parametros y multi-judge).
	- Campos principales: `models`, `temperatures`, `top_p`, `top_k`, `repetitions`, `dataset_path`, `eval_split`, `output_path`, `judges`.

- `datasets/`
	- Rol: fuente de tareas en JSONL (una tarea por linea).
	- `slm_tasks_ptbr.jsonl`: dataset inicial estratificado en 5 familias.
	- Schema practico por item: `id`, `split`, `task_type`, `difficulty`, `prompt`, `reference`.

- `scripts/`
	- Rol: interfaz CLI para ejecutar el pipeline end-to-end.
	- `run_benchmark.py`: carga config, opcionalmente revisa Ollama local, ejecuta trials y guarda JSONL crudo.
	- `analyze_results.py`: ejecuta analisis estadistico y genera reporte markdown.
	- `prepare_human_eval.py`: crea paquete ciego para anotacion humana con overlap controlado.
	- `agreement_report.py`: mide concordancia entre evaluadores (correlaciones + kappa ponderado).

- `src/slm_benchmark/`
	- Rol: logica nucleo del benchmark.
	- `config.py`: parseo/validacion de YAML en `BenchmarkConfig`.
	- `dataset.py`: carga robusta de dataset JSONL en objetos `DatasetItem`.
	- `clients.py`: cliente Ollama (healthcheck, listado de modelos, generacion de respuesta).
	- `judges.py`: evaluadores automaticos (heuristico y juez LLM via Ollama).
	- `runner.py`: genera combinaciones de trial, aleatoriza orden, ejecuta, agrega score por mediana, persiste resultados.
	- `analysis.py`: resumen por modelo con metricas operativas, IC bootstrap, Welch t-test con Cohen's d y OLS robusto por item.
	- `human_eval.py`: muestreo estratificado y distribucion ciega de tareas para humanos.
	- `agreement.py`: reporte de concordancia inter-evaluador.

- `results/`
	- Rol: artefactos generados por la ejecucion.
	- Tipicos: `raw_benchmark*.jsonl`, `report.md`, `human_assignment.csv`, `human_agreement.md`.
	- Buena practica: crear subcarpetas por release (`results/release-vX.Y.Z/`).

- `docs/`
	- Rol: gobernanza metodologica.
	- `human_eval_rubric.es.md`: criterios de scoring humano (0 a 10 por dimension).
	- `release_protocol.es.md`: checklist cientifico para releases reproducibles.

- `templates/`
	- Rol: estandar de comunicacion tecnica/cientifica.
	- `report_template.md`: template de reporte tecnico.
	- `paper_outline.md`: estructura base para paper.

- Archivos raiz
	- `README.md`: guia operativa del proyecto.
	- `pyproject.toml`: empaquetado Python + dependencias.
	- `VERSION` y `CHANGELOG.md`: control de version e historial.
	- `LICENSE`: licenciamiento.

### 5.3 Como se conectan los modulos internamente

1. `scripts/run_benchmark.py` llama `load_config` (`config.py`).
2. `runner.py` llama `load_jsonl_dataset` (`dataset.py`) y crea los trials.
3. Cada trial usa `OllamaClient.generate` (`clients.py`).
4. La respuesta es evaluada por `build_judges` (`judges.py`) y agregada por mediana.
5. El resultado se escribe en JSONL en `results/`.
6. `scripts/analyze_results.py` llama `analysis.py` y genera estadisticas en markdown.
7. Si hay evaluacion humana, `human_eval.py` y `agreement.py` cierran el ciclo.

## 6. Flujo de contribucion y nuevas funcionalidades

Para evolucionar un repositorio que ya existe, usa un flujo de rama de funcionalidad + Pull Request:

```bash
git checkout -b feat/nombre-de-funcionalidad
# implementa tus cambios y ejecuta las pruebas necesarias
git add .
git commit -m "feat: descripcion corta de la funcionalidad"
git push -u origin feat/nombre-de-funcionalidad
```

Despues, abre un Pull Request en GitHub para revisar e integrar en `main`.

Checklist recomendado antes del PR:
1. ejecutar el smoke test local (`configs/benchmark_local_smoke.yaml`);
2. actualizar `CHANGELOG.md` cuando el cambio sea relevante;
3. validar si `VERSION` debe incrementarse para release.

## 7. Dataset estratificado

Este proyecto ya incluye un dataset inicial con 40 tareas estratificadas en 5 familias:
- code_explanation
- bug_detection
- refactoring
- test_generation
- security_performance

Para el protocolo academico 0.2.0, el dataset tiene split materializado en cada item:
- `train`: 24 tareas
- `dev`: 8 tareas
- `test`: 8 tareas

Archivo:
- datasets/slm_tasks_ptbr.jsonl

## 8. Evaluacion humana ciega

Los comandos de preparacion y concordancia ya aparecen en onboarding 2.1 (pasos 4 y 5).

Rubrica:
- docs/human_eval_rubric.es.md

### 8.1 Validaciones de entrada (actual)

El flujo de evaluacion humana ahora falla temprano con mensaje claro cuando:
- `--evaluators` esta vacio;
- `--sample-size` es menor o igual a cero;
- `--overlap-rate` esta fuera del intervalo `[0, 1]`;
- el JSONL de entrada no tiene filas con `valid_response = true`.

## 9. Protocolo cientifico de release

Versionado y gobernanza metodologica:
- VERSION
- CHANGELOG.md
- docs/release_protocol.es.md

Templates para difusion:
- templates/report_template.md
- templates/paper_outline.md

