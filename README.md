# slm-benchmark-kit

Framework reutilizavel para benchmark de SLMs (Small Language Models), desacoplado do projeto Sagui.

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

## 3. Rodar benchmark

```bash
python scripts/run_benchmark.py --config configs/benchmark_ollama.yaml
```

Resultado bruto (JSONL):
- `results/raw_benchmark.jsonl`

## 4. Gerar relatorio

```bash
python scripts/analyze_results.py --input results/raw_benchmark.jsonl --output results/report.md
```

## 5. Estrutura

- `datasets/`: tarefas em JSONL
- `configs/`: configuracoes de experimento
- `scripts/`: entrypoints CLI
- `src/slm_benchmark/runner.py`: execucao e coleta
- `src/slm_benchmark/analysis.py`: estatistica robusta

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

## 7. Proximos passos recomendados

- ampliar dataset para 30-50 tarefas estratificadas;
- adicionar avaliacao humana em amostra cega;
- calcular concordancia entre avaliadores;
- adicionar testes de robustez (parafrase e ruido de prompt).
