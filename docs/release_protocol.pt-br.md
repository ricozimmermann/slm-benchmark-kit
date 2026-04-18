# Protocolo de Lançamento Científico

Este protocolo garante que cada release do benchmark seja reproduzível e auditável.

## 1) Regras de versionamento

- As versões do benchmark seguem SemVer: MAJOR.MINOR.PATCH.
- MAJOR: mudanças de protocolo com quebra de compatibilidade (schema do dataset, lógica de score, desenho dos trials).
- MINOR: adições metodológicas sem quebra de compatibilidade (novas tarefas, novas métricas).
- PATCH: correções de bugs que não alteram conclusões científicas.

Arquivos que devem ser atualizados em cada release:
- VERSION
- CHANGELOG.md
- configs/benchmark_ollama.yaml (se a configuração mudar)
- datasets/slm_tasks_ptbr.jsonl (se o dataset mudar)
- checksum do dataset (SHA256) em metadata.json

## 2) Artefatos obrigatórios da release

Para cada tag de release, publique:
- JSONL bruto do benchmark.
- Relatório markdown resumido.
- Template de human eval usado.
- Relatório de agreement humano.
- YAML exato da configuração em runtime.
- SHA256 do dataset e a política de split usada na avaliação.
- Hash do commit e metadados da plataforma.

## 3) Guardrails metodológicos

- Mantenha o split de holdout intacto durante o tuning.
- Defina `eval_split: test` para o reporte científico.
- Use seed fixa em cada release.
- Registre os IDs dos modelos exatamente como executados no Ollama.
- Mantenha o arquivo de mapeamento cego privado durante a avaliação humana.
- Não altere arquivos humanos já pontuados após a análise de agreement.

## 4) Checklist de release

- [ ] Atualizar o VERSION.
- [ ] Acrescentar no CHANGELOG as mudanças de método/dados.
- [ ] Rodar o benchmark com a configuração de release.
- [ ] Gerar o relatório estatístico.
- [ ] Confirmar inferência robusta (cluster por item quando disponível).
- [ ] Preparar a amostra cega para avaliação humana.
- [ ] Coletar scores humanos e executar o relatório de agreement.
- [ ] Arquivar os artefatos em `results/release-vX.Y.Z/`.
- [ ] Marcar a release no git: `vX.Y.Z`.

## 5) Estrutura sugerida de pasta da release

results/release-vX.Y.Z/
- raw_benchmark.jsonl
- report.md
- config.yaml
- human_assignment.csv
- human_key_private.csv
- human_scored.csv
- human_agreement.md
- metadata.json