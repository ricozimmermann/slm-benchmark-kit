# Quick Guide — SLM Benchmark Kit

Este guia reúne informações práticas de uso do SLM Benchmark Kit: configuração, comandos principais, exemplos de tarefas e rastreabilidade dos arquivos de avaliação humana.

---

## A. Guia Prático de Uso do Benchmark

Esta seção resume como executar o SLM Benchmark Kit no dia a dia, incluindo como trocar modelos e como disparar os comandos principais.

### A.1 Configuração Principal em `benchmark_ollama.yaml`

No arquivo `configs/benchmark_ollama.yaml`, o bloco `models` define os modelos avaliados. Para substituir os modelos, basta editar a lista.

Exemplo atual:
```yaml
models:
  - "deepseek-coder:1.3b"
  - "qwen2.5-coder:1.5b"
```

Exemplo de substituição (apenas ilustrativo):
```yaml
models:
  - "qwen2.5-coder:3b"
  - "phi3:mini"
```

Além de `models`, os campos `temperatures`, `top_p`, `top_k`, `repetitions`, `eval_split` e `output_path` controlam a malha experimental e o local de saída.

### A.2 Comandos Principais de Execução

Fluxo local recomendado:
```bash
python scripts/run_benchmark.py --config configs/benchmark_local_smoke.yaml --check-local
python scripts/run_benchmark.py --config configs/benchmark_ollama.yaml
python scripts/analyze_results.py --input results/raw_benchmark.jsonl --output results/report.md
```

Fluxo opcional de avaliação humana:
```bash
python scripts/prepare_human_eval.py --input results/raw_benchmark.jsonl \
  --assignment results/human_assignment.csv \
  --key results/human_key_private.csv \
  --evaluators eval01 eval02 eval03 \
  --sample-size 120 --overlap-rate 0.25

python scripts/agreement_report.py --input results/human_assignment_scored.csv \
  --key results/human_key_private.csv \
  --output results/human_agreement.md
```

Em ambiente Docker (quando aplicável), o fluxo equivalente utiliza:
```bash
docker compose up -d ollama
docker compose run --rm ollama-pull
docker compose run --rm benchmark
docker compose run --rm analysis
```

---

## B. Exemplos de Tarefas e Respostas (5 Casos)

Esta seção apresenta cinco tarefas reais de `datasets/slm_tasks_ptbr.jsonl` e os respectivos trechos de resposta registrados em `results/raw_benchmark.jsonl`. Os casos foram rastreados via IDs cegos usados também na avaliação humana.

### B.1 Caso 1: BLIND-0001 (task-015, bug_detection)

**Tarefa:**
```
Encontre o problema: const n = Number(input); if(n){ process(n); }
```

**Resposta (trecho):**
```
O codigo ... verifica se os dados inseridos no input sao valido ...
O problema e que voce esta utilizando a instrucao if com um operador
de comparacao ...
```

**Explicação:** o modelo identifica a necessidade de validação de entrada, mas mistura conceitos (operadores lógicos versus condição de truthiness). Isso explica o gap entre score agregado automático alto e avaliação humana moderada.

### B.2 Caso 2: BLIND-0009 (task-016, bug_detection)

**Tarefa:**
```
Encontre o problema de memoria: const cache=[];
setInterval(()=>cache.push(new Array(1000000).fill('x')), 1000);
```

**Resposta (trecho):**
```
... insere novo array com 1.000.000 posicoes 'x' dentro de cache[] e
nunca remove os elementos inseridos anteriormente ...
```

**Explicação:** a resposta acerta o diagnóstico principal (crescimento sem limite do cache), mas a recomendação de solução aparece genérica em parte do texto.

### B.3 Caso 3: BLIND-0017 (task-008, code_explanation)

**Tarefa:**
```
Explique este uso de optional chaining:
const cidade = user?.endereco?.cidade ?? 'desconhecida';
```

**Resposta (trecho):**
```
Este codigo ... utiliza a sintaxe do Optional Chaining (`?.`) ...
... permitindo nao disparar erros caso alguma referencia falhe ...
```

**Explicação:** embora explique a intenção geral de segurança no acesso aninhado, a formulação traz imprecisões conceituais, refletindo score agregado automático menor neste caso.

### B.4 Caso 4: BLIND-0025 (task-007, code_explanation)

**Tarefa:**
```
Explique closures neste codigo:
function makeCounter(){ let c=0; return () => ++c; }
```

**Resposta (trecho):**
```
Este codigo JavaScript es un ejemplo de closure ...
Un closure es una funcion que puede acceder a variables fuera de su
scope ...
```

**Explicação:** a resposta descreve corretamente o núcleo de closures, porém apresenta _code-switching_ (português-espanhol), ruído linguístico e pouca precisão aplicada ao snippet específico. Esse fenômeno pode interagir com a dimensão de clareza no julgamento humano, reduzindo `score_clarity` mesmo quando a correção técnica central está presente.

### B.5 Caso 5: BLIND-0057 (task-032, test_generation)

**Tarefa:**
```
Crie testes para middleware de autenticacao que depende de header Bearer.
```

**Resposta (trecho):**
```
... voce precisa primeiro configurar um servidor NodeJS para simular
requisicoes HTTP ... instalar supertest ...
```

**Explicação:** a saída oferece caminho inicial útil (ferramentas e estrutura), mas não cobre de modo completo casos de token ausente, inválido, válido e expirado, previstos na referência da tarefa.

---

## C. Cinco Resultados em `human_key_private`, `human_assignment` e `human_assignment_scored`

Os mesmos cinco casos acima são apresentados com suas entradas nos três arquivos de avaliação humana.

### C.1 Resultados em `human_key_private.csv`

| blind_id   | item_id  | modelo               | score_aggregated | latência (ms) |
|------------|----------|----------------------|-----------------|--------------|
| BLIND-0001 | task-015 | deepseek-coder:1.3b  | 8.00            | 9501         |
| BLIND-0009 | task-016 | deepseek-coder:1.3b  | 8.00            | 9064         |
| BLIND-0017 | task-008 | deepseek-coder:1.3b  | 5.75            | 8683         |
| BLIND-0025 | task-007 | deepseek-coder:1.3b  | 8.00            | 9785         |
| BLIND-0057 | task-032 | deepseek-coder:1.3b  | 7.00            | 11425        |

### C.2 Resultados em `human_assignment.csv`

| blind_id   | trecho da resposta cega |
|------------|------------------------|
| BLIND-0001 | Resposta sobre validação de `Number(input)` e tratamento condicional de entrada. |
| BLIND-0009 | Resposta sobre crescimento infinito de `cache` com `setInterval` e risco de memória. |
| BLIND-0017 | Resposta sobre _optional chaining_ e fallback para valor desconhecido. |
| BLIND-0025 | Resposta sobre conceito de closures e captura de escopo léxico. |
| BLIND-0057 | Resposta sobre criação de testes para middleware Bearer com `supertest`/NodeJS. |

### C.3 Resultados em `human_assignment_scored.csv`

| blind_id   | overall | técnico | completude | acionável | clareza | formato | notas |
|------------|---------|---------|-----------|-----------|---------|---------|-------|
| BLIND-0001 | 7.5     | 8.4     | 7.0       | 8.0       | 8.0     | 8.0     | Boa explicação técnica |
| BLIND-0009 | 7.6     | 7.8     | 8.0       | 8.0       | 7.0     | 6.0     | Poderia ser mais específico |
| BLIND-0017 | 7.5     | 6.0     | 7.0       | 8.0       | 8.0     | 10.0    | Boa explicação técnica |
| BLIND-0025 | 6.1     | 7.0     | 7.0       | 4.0       | 6.0     | 6.0     | Exemplos de código ajudam na compreensão |
| BLIND-0057 | 7.1     | 7.5     | 6.0       | 7.0       | 7.0     | 8.0     | Explicação clara e direta |

### C.4 Leitura Integrada dos Cinco Casos

Os cinco exemplos mostram três pontos metodológicos relevantes.

1. O `score_aggregated` automático pode divergir de `score_overall` humano quando a resposta parece estruturalmente correta, mas falha em precisão ou aplicabilidade.
2. Notas qualitativas dos avaliadores (campo `notes`) ajudam a explicar onde o texto falha para uso prático.
3. A rastreabilidade por `blind_id` permite auditar de ponta a ponta o fluxo entre resposta bruta, pacote cego e pontuação final.
