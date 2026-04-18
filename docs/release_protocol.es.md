# Protocolo Científico de Publicación

Este protocolo garantiza que cada release del benchmark sea reproducible y auditable.

## 1) Reglas de versionado

- Las versiones del benchmark siguen SemVer: MAJOR.MINOR.PATCH.
- MAJOR: cambios de protocolo incompatibles (schema del dataset, lógica de scoring, diseño de trials).
- MINOR: adiciones metodológicas compatibles (nuevas tareas, nuevas métricas).
- PATCH: correcciones de errores que no cambian las conclusiones científicas.

Archivos que deben actualizarse en cada release:
- VERSION
- CHANGELOG.md
- configs/benchmark_ollama.yaml (si cambia la configuración)
- datasets/slm_tasks_ptbr.jsonl (si cambia el dataset)
- checksum del dataset (SHA256) en metadata.json

## 2) Artefactos obligatorios de la release

Para cada etiqueta de release, publique:
- JSONL bruto del benchmark.
- Informe resumen en markdown.
- Plantilla de evaluación humana utilizada.
- Informe de acuerdo humano.
- YAML exacto de configuración en tiempo de ejecución.
- SHA256 del dataset y la política de split usada en la evaluación.
- Hash del commit de git y metadatos de la plataforma.

## 3) Guardrails metodológicos

- Mantenga intacto el split de evaluación holdout durante el tuning.
- Establezca `eval_split: test` para el reporte científico.
- Use una semilla fija para cada release.
- Registre los IDs de modelo exactamente como se ejecutaron en Ollama.
- Mantenga privado el archivo de mapeo ciego durante la evaluación humana.
- No modifique los archivos humanos ya puntuados después del análisis de acuerdo.

## 4) Lista de verificación de release

- [ ] Incrementar VERSION.
- [ ] Agregar al CHANGELOG los cambios de método/datos.
- [ ] Ejecutar el benchmark con la configuración de release.
- [ ] Generar el informe estadístico.
- [ ] Confirmar inferencia robusta (cluster por item cuando esté disponible).
- [ ] Preparar la muestra ciega de evaluación humana.
- [ ] Recopilar scores humanos y ejecutar el informe de acuerdo.
- [ ] Archivar los artefactos en `results/release-vX.Y.Z/`.
- [ ] Marcar la release en git: `vX.Y.Z`.

## 5) Estructura sugerida de carpeta de release

results/release-vX.Y.Z/
- raw_benchmark.jsonl
- report.md
- config.yaml
- human_assignment.csv
- human_key_private.csv
- human_scored.csv
- human_agreement.md
- metadata.json