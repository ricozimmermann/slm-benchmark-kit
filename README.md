# slm-benchmark-kit

Framework reutilizavel para benchmark de SLMs (Small Language Models).

Objetivo:
- comparar modelos e parametros com reproducibilidade;
- suportar multiplos projetos com o mesmo protocolo;
- produzir analise estatistica real (Welch t-test, OLS, IC bootstrap).

## 1. Estrategia metodologica

Este projeto ja incorpora melhorias essenciais de benchmark:
- random seed fixa para reproducibilidade;
- randomizacao da ordem de trials;
- repeticoes por combinacao de parametros;
- multi-judge com agregacao por mediana;
- analise estatistica com scipy/statsmodels (sem p-value heuristico).

## 2. Instalacao

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 2.1 Onboarding rapido (ordem recomendada)

Se for sua primeira vez no projeto, siga esta ordem:

1. Rode um smoke test local para validar ambiente e Ollama:

```bash
python scripts/run_benchmark.py --config configs/benchmark_local_smoke.yaml --check-local
```

2. Rode o benchmark completo (ou o seu config customizado):

```bash
python scripts/run_benchmark.py --config configs/benchmark_ollama.yaml
```

3. Gere o relatorio estatistico:

```bash
python scripts/analyze_results.py --input results/raw_benchmark.jsonl --output results/report.md
```

4. (Opcional) Prepare avaliacao humana cega:

```bash
python scripts/prepare_human_eval.py \
	--input results/raw_benchmark.jsonl \
	--assignment results/human_assignment.csv \
	--key results/human_key_private.csv \
	--evaluators eval01 eval02 eval03 \
	--sample-size 120 \
	--overlap-rate 0.25
```

5. (Opcional) Gere concordancia entre avaliadores:

```bash
python scripts/agreement_report.py \
	--input results/human_assignment_scored.csv \
	--output results/human_agreement.md
```

## 3. Rodar benchmark

```bash
python scripts/run_benchmark.py --config configs/benchmark_ollama.yaml
```

### 3.1 Teste local rapido (SLM em Ollama)

Use a configuracao smoke para validar pipeline localmente com poucas amostras:

```bash
python scripts/run_benchmark.py --config configs/benchmark_local_smoke.yaml --check-local
```

Isso faz:
- checagem de conectividade com Ollama local;
- aviso se o modelo configurado nao estiver baixado;
- execucao com `trial_limit` pequeno para teste rapido.

Resultado bruto (JSONL):
- `results/raw_benchmark.jsonl`

## 4. Gerar relatorio

```bash
python scripts/analyze_results.py --input results/raw_benchmark.jsonl --output results/report.md
```

## 5. Arquitetura do projeto (didatica)

### 5.1 Mapa visual do fluxo

```text
datasets/*.jsonl
	 -> configs/*.yaml
	 -> scripts/run_benchmark.py
	 -> src/slm_benchmark/runner.py
	 -> results/raw_benchmark*.jsonl
	 -> scripts/analyze_results.py
	 -> src/slm_benchmark/analysis.py
	 -> results/report.md

Opcional (avaliacao humana):
results/raw_benchmark*.jsonl
	 -> scripts/prepare_human_eval.py
	 -> results/human_assignment.csv + results/human_key_private.csv
	 -> (avaliadores preenchem scores)
	 -> scripts/agreement_report.py
	 -> results/human_agreement.md
```

### 5.2 Pasta por pasta

- `configs/`
	- Papel: definir o experimento sem alterar codigo.
	- `benchmark_local_smoke.yaml`: validacao rapida (poucas combinacoes, `trial_limit` baixo).
	- `benchmark_ollama.yaml`: benchmark principal (mais modelos/parametros e multi-judge).
	- Campos principais: `models`, `temperatures`, `top_p`, `top_k`, `repetitions`, `dataset_path`, `output_path`, `judges`.

- `datasets/`
	- Papel: fonte de tarefas em JSONL (uma tarefa por linha).
	- `slm_tasks_ptbr.jsonl`: dataset inicial estratificado em 5 familias.
	- Schema pratico por item: `id`, `task_type`, `difficulty`, `prompt`, `reference`.

- `scripts/`
	- Papel: interface CLI para operar o pipeline ponta a ponta.
	- `run_benchmark.py`: carrega config, opcionalmente checa Ollama local, executa trials e salva JSONL bruto.
	- `analyze_results.py`: roda analise estatistica e gera markdown de resultados.
	- `prepare_human_eval.py`: cria pacote cego para anotacao humana com overlap controlado.
	- `agreement_report.py`: mede concordancia entre avaliadores (correlacoes + kappa ponderado).

- `src/slm_benchmark/`
	- Papel: nucleo de logica do benchmark.
	- `config.py`: parse/validacao de YAML em `BenchmarkConfig`.
	- `dataset.py`: leitura robusta de dataset JSONL para objetos `DatasetItem`.
	- `clients.py`: cliente Ollama (healthcheck, listagem de modelos, geracao de resposta).
	- `judges.py`: avaliadores automaticos (heuristico e judge por LLM via Ollama).
	- `runner.py`: gera combinacoes de trial, randomiza ordem, executa, agrega score por mediana, persiste resultados.
	- `analysis.py`: resumo por modelo, IC bootstrap, Welch t-test e regressao OLS.
	- `human_eval.py`: amostragem estratificada e distribuicao de tarefas cegas para humanos.
	- `agreement.py`: relatorio de concordancia interavaliador.

- `results/`
	- Papel: artefatos gerados pela execucao.
	- Tipicos: `raw_benchmark*.jsonl`, `report.md`, `human_assignment.csv`, `human_agreement.md`.
	- Boa pratica: criar subpastas por release (`results/release-vX.Y.Z/`).

- `docs/`
	- Papel: governanca metodologica.
	- `human_eval_rubric.md`: criterios de scoring humano (0 a 10 por dimensao).
	- `release_protocol.md`: checklist cientifico para releases reproduziveis.

- `templates/`
	- Papel: padrao de comunicacao tecnica/cientifica.
	- `report_template.md`: template de relatorio tecnico.
	- `paper_outline.md`: estrutura-base para artigo.

- Arquivos de raiz
	- `README.md`: guia operacional do projeto.
	- `pyproject.toml`: empacotamento Python + dependencias.
	- `VERSION` e `CHANGELOG.md`: controle de versao e historico.
	- `LICENSE`: licenciamento.

### 5.3 Como os modulos se conectam internamente

1. `scripts/run_benchmark.py` chama `load_config` (`config.py`).
2. `runner.py` chama `load_jsonl_dataset` (`dataset.py`) e monta os trials.
3. Cada trial usa `OllamaClient.generate` (`clients.py`).
4. A resposta e avaliada por `build_judges` (`judges.py`) e agregada por mediana.
5. O resultado vai para JSONL em `results/`.
6. `scripts/analyze_results.py` chama `analysis.py` para estatistica e gera markdown.
7. Se houver avaliacao humana, `human_eval.py` e `agreement.py` fecham o ciclo.

## 6. Publicar como repositorio publico

No diretorio do projeto:

```bash
git init
git add .
git commit -m "feat: initial slm benchmark kit"
```

Se tiver GitHub CLI (`gh`) autenticado:

```bash
gh repo create slm-benchmark-kit --public --source . --remote origin --push
```

Se preferir criar manualmente no GitHub:
1. crie um repositorio publico vazio;
2. rode:

```bash
git remote add origin https://github.com/<seu-usuario>/slm-benchmark-kit.git
git branch -M main
git push -u origin main
```

## 7. Dataset estratificado

Este projeto ja inclui um dataset inicial com 40 tarefas estratificadas em 5 familias:
- code_explanation
- bug_detection
- refactoring
- test_generation
- security_performance

Arquivo:
- datasets/slm_tasks_ptbr.jsonl

## 8. Avaliacao humana cega

Gerar pacote de anotacao cega com overlap para medicao de concordancia:

```bash
python scripts/prepare_human_eval.py \
	--input results/raw_benchmark.jsonl \
	--assignment results/human_assignment.csv \
	--key results/human_key_private.csv \
	--evaluators eval01 eval02 eval03 \
	--sample-size 120 \
	--overlap-rate 0.25
```

Depois de preencher score_overall e demais campos no CSV de assignment:

```bash
python scripts/agreement_report.py \
	--input results/human_assignment_scored.csv \
	--output results/human_agreement.md
```

Rubrica:
- docs/human_eval_rubric.md

## 9. Protocolo de release cientifico

Versionamento e governanca metodologica:
- VERSION
- CHANGELOG.md
- docs/release_protocol.md

Templates para divulgacao:
- templates/report_template.md
- templates/paper_outline.md

## 10. Publicacao publica no GitHub

Se ainda nao tiver remoto configurado:

```bash
git remote add origin https://github.com/<seu-usuario>/slm-benchmark-kit.git
git branch -M main
git push -u origin main
```
