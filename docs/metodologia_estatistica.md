# Metodologia Estatistica do SLM Benchmark Kit

## 0. Justificativa cientifica: por que SLMs e por que benchmark

O interesse por Small Language Models (SLMs) e consistente com uma mudanca de foco na IA aplicada: sair de solucoes exclusivamente baseadas em escala maxima e avançar para modelos eficientes, executaveis em cenarios com restricoes reais de custo, energia, latencia e privacidade.

No contexto pratico, SLMs sao particularmente relevantes porque:
- reduzem requisitos de memoria e processamento, viabilizando uso em edge/mobile e ambientes com infraestrutura limitada;
- permitem processamento local/offline, com menor dependencia de nuvem e menor exposicao de dados sensiveis;
- favorecem democratizacao tecnologica, ampliando acesso para instituicoes com menor capacidade computacional;
- podem entregar desempenho competitivo em dominios especificos quando bem ajustados e avaliados.

Entretanto, a maior eficiencia de um SLM nao implica automaticamente melhor desempenho util. Por isso, a comparacao entre modelos precisa ser feita sob protocolo experimental controlado, com inferencia estatistica explicita e metricas operacionais complementares.

Assim, a justificativa para benchmark neste projeto e dupla:
1. metodologica: medir diferencas de desempenho com controle de incerteza e de confundimento;
2. aplicada: identificar o melhor equilibrio entre qualidade, estabilidade, latencia e custo computacional para cenarios reais.

Em sintese, SLMs sao estrategicos por eficiencia e acessibilidade; benchmark reprodutivel e estrategico por confiabilidade da decisao tecnica.

## 1. Resumo executivo

Este documento formaliza o conteudo estatistico utilizado no projeto `slm-benchmark-kit` para comparar SLMs com foco em:
- estimacao com incerteza explicita;
- comparacao inferencial entre modelos;
- controle de covariaveis experimentais;
- diagnostico de qualidade do sistema de avaliacao automatica (judges).

A pipeline atual combina:
- estatistica descritiva por modelo;
- intervalo de confianca bootstrap para media de score;
- teste t de Welch para comparacao entre dois modelos;
- tamanho de efeito de Cohen (d);
- regressao OLS com erros robustos (cluster por item quando disponivel; HC3 como fallback);
- metricas de concordancia e saude dos juizes.

## 2. Unidade de analise e desenho experimental

## 2.1 Unidade observacional

A unidade observacional e o `trial` registrado no JSONL de resultados, contendo:
- identificacao do modelo (`model`);
- identificacao do item (`item_id`), tipo de tarefa e dificuldade;
- hiperparametros de geracao (`temperature`, `top_p`, `top_k`);
- repeticao (`repetition`);
- score agregado (`score_aggregated`) quando ha pelo menos um juiz valido.

## 2.2 Definicoes operacionais

- `valid_response`: no contexto do pipeline atual, significa que pelo menos um juiz retornou score valido (`judge_valid_count > 0`).
- `score_aggregated`: mediana dos scores validos dos juizes para o trial.
- `error`: erro de geracao (por exemplo, timeout). Quando presente, o trial tende a ficar sem score agregado.

## 2.3 Estrategias para reduzir vies

O desenho implementado no benchmark contempla:
- seed fixa para reprodutibilidade;
- randomizacao da ordem de execucao dos trials;
- repeticoes por combinacao de parametros;
- separacao por split de avaliacao (`eval_split`) para evitar leakage entre tuning e teste;
- avaliacao por multiplos juizes com agregacao robusta por mediana.

## 3. Variaveis e metricas reportadas

## 3.1 Performance principal

Para cada modelo, sao reportados:
- `n`: total de trials;
- `n_scored`: quantidade de trials com `score_aggregated` numerico;
- media, desvio-padrao amostral e mediana de `score_aggregated`;
- intervalo de confianca bootstrap de 95% da media (`ci95_low`, `ci95_high`).

## 3.2 Confiabilidade operacional

Tambem sao reportados:
- `valid_rate` (media de `valid_response`);
- `error_rate` (proporcao de erros nao vazios);
- `timeout_rate` (proporcao de erros contendo "timeout");
- `judge_all_failed_rate` (proporcao de trials com `judge_valid_count <= 0`);
- `latency_p50_ms` e `latency_p95_ms`.

Essas metricas evitam conclusoes baseadas apenas em score medio e evidenciam custo/estabilidade.

## 4. Metodos de inferencia estatistica

## 4.1 Intervalo de confianca bootstrap para a media

Para cada modelo, o IC de 95% da media e estimado por bootstrap percentile:
1. amostrar com reposicao, `n_boot = 5000`, do vetor de scores validos;
2. calcular a media em cada reamostragem;
3. usar os percentis 2.5% e 97.5% da distribuicao bootstrap.

Configuracao atual:
- `seed = 42` no bootstrap.

Interpretacao:
- IC mais estreito indica maior precisao da estimativa media;
- sobreposicao de ICs nao substitui teste formal de hipotese.

## 4.2 Teste t de Welch (comparacao entre dois modelos)

Quando ha exatamente dois modelos com ao menos 2 observacoes validas por grupo, aplica-se teste t de Welch (variancias desiguais):

- Hipotese nula: medias populacionais iguais.
- Hipotese alternativa: medias populacionais diferentes.

O relatorio inclui:
- `t_stat`;
- `p_value`;
- indicador `significant_0_05`.

Observacao de escopo:
- a implementacao atual executa Welch apenas no caso de 2 modelos; para mais de 2 modelos, o teste e pulado com aviso.

## 4.3 Tamanho de efeito (Cohen's d)

Junto ao Welch, e calculado Cohen's d para quantificar magnitude do efeito:

- d ~ 0.2: efeito pequeno (regra pratica);
- d ~ 0.5: efeito medio;
- d ~ 0.8: efeito grande.

Boas praticas:
- interpretar `p_value` em conjunto com `d`;
- reportar tambem diferenca de medias (`mean_a - mean_b`) e contexto aplicado.

## 4.4 Regressao OLS com erros robustos

Para controlar confundimento de configuracao e composicao de tarefas, o projeto ajusta:

`score_aggregated ~ C(model) + temperature + top_p + top_k + C(task_type)`

Detalhes:
- `model` e `task_type` entram como fatores categoricos;
- o ajuste ocorre somente se houver colunas obrigatorias e ao menos 8 observacoes validas;
- se `item_id` possuir mais de um valor unico, usa-se covariancia cluster-robust por item;
- em caso contrario (ou falha), usa-se HC3.

Interpretacao recomendada:
- coeficientes de `C(model)` representam diferencas condicionais (controladas por covariaveis);
- erros robustos reduzem sensibilidade a heterocedasticidade;
- cluster por item ajuda a lidar com dependencia intra-item entre repeticoes.

## 5. Qualidade do sistema de avaliacao automatica (judges)

## 5.1 Concordancia par-a-par

A partir de `judge_scores`, o sistema calcula por par de juizes:
- Spearman;
- Kendall tau;
- Pearson;
- MAE (erro absoluto medio);
- `n_overlap` (amostras com score valido em ambos).

Tratamento de casos limite:
- se `n_overlap < 3`, correlacoes sao reportadas como `NaN`;
- se uma serie for constante, correlacoes sao `NaN` e MAE permanece calculado.

## 5.2 Saude dos juizes

A partir de `judge_rationales`, o sistema estima por juiz:
- `valid_rate`;
- `parse_error_rate`;
- `judge_error_rate`;
- `parse_fallback_rate`.

Essas metricas permitem diferenciar:
- divergencia de criterio (baixa concordancia);
- falha tecnica de parse/infra (baixa saude operacional).

## 6. Ameacas a validade e limitacoes

## 6.1 Validade interna

- Dependencia entre trials: repeticoes do mesmo `item_id` nao sao independentes em sentido estrito (mitigado parcialmente com cluster-robust no OLS).
- Selecao por respostas validas: analises de score usam `score_aggregated` nao nulo; diferencas de taxa de falha entre modelos podem enviesar comparacoes de media.

## 6.2 Validade externa

- Resultados dependem do dataset atual (`slm_tasks_ptbr.jsonl`), idioma e distribuicao de tarefas.
- Generalizacao para outros dominios requer replicacao com novos conjuntos estratificados.

## 6.3 Validade de construto

- O score agregado depende da rubrica dos juizes (heuristico/SLM ou LLM) e da robustez do parse.
- Concordancia alta nao implica validade semantica total; por isso recomenda-se avaliacao humana cega complementar.

## 7. Reprodutibilidade e reporte cientifico

Para reporte auditavel, recomenda-se sempre publicar:
- JSONL bruto;
- relatorio markdown gerado pelo pipeline;
- YAML de configuracao exato;
- hash de commit;
- versao do dataset e checksum;
- versao do protocolo (`VERSION` + `CHANGELOG.md`).

Checklist operacional complementar:
1. executar benchmark com `eval_split: test` para conclusoes finais;
2. registrar falhas (`error_rate`, `timeout_rate`) junto aos scores;
3. reportar efeito + incerteza (diferenca de medias, IC, `p_value`, `d`);
4. evitar conclusoes com base exclusiva em limiar de 0.05.

## 8. Diretrizes de interpretacao

Ao comparar modelos, priorize esta ordem:
1. viabilidade operacional (valid_rate, error_rate, timeout_rate, latencia);
2. estimativa central e incerteza (media + IC bootstrap);
3. evidencia inferencial (Welch + tamanho de efeito);
4. analise ajustada (OLS robusto) para robustez da conclusao.

Um modelo com media maior, mas alta taxa de falha, pode ser pior em cenario real que um modelo com media ligeiramente menor e estabilidade superior.

## 9. Referencias metodologicas (ABNT)

- CAMERON, A. Colin; MILLER, Douglas L. A Practitioner's Guide to Cluster-Robust Inference. Journal of Human Resources, v. 50, n. 2, p. 317-372, 2015. DOI: https://doi.org/10.3368/jhr.50.2.317.
- COHEN, Jacob. Statistical Power Analysis for the Behavioral Sciences. 2. ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988. ISBN: 9780805802832.
- EFRON, Bradley; TIBSHIRANI, Robert J. An Introduction to the Bootstrap. New York: Chapman & Hall, 1993. ISBN: 9780412042317.
- MACKINNON, James G.; WHITE, Halbert. Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties. Journal of Econometrics, v. 29, n. 3, p. 305-325, 1985. DOI: https://doi.org/10.1016/0304-4076(85)90158-7.
- WELCH, B. L. The Generalization of Student's Problem when Several Different Population Variances are Involved. Biometrika, v. 34, n. 1-2, p. 28-35, 1947. DOI: https://doi.org/10.2307/2332510.

### 9.1 Nota de rastreabilidade

As referencias desta secao foram verificadas por metadados bibliograficos em fontes externas:
- artigos: Crossref (titulo, autores, periodico, volume, numero, paginas e DOI);
- livros: catalogos ISBN (Open Library/Google Books) para autores, edicao, editora e ano.

Data da verificacao: 2026-04-15.

### 9.2 Evidencia de uso em repositorios brasileiros

Para responder a possivel duvida sobre "atualidade" dos metodos, foi realizada consulta adicional automatizada em repositorios institucionais brasileiros. O objetivo nao foi substituir as referencias fundacionais, mas verificar uso recente em teses/dissertacoes.

Resumo dos retornos observados:
- UFMG ([repositorio.ufmg.br](https://repositorio.ufmg.br/)): busca por bootstrap com 580 resultados; busca por welch com 394 resultados.
- UFPE ([repositorio.ufpe.br](https://repositorio.ufpe.br/)): busca por bootstrap com 1196 resultados; busca por welch com 766 resultados.
- UFF ([app.uff.br/riuff](https://app.uff.br/riuff/)): busca por bootstrap com 34 resultados; busca por welch com 35 resultados.

Exemplos de itens retornados nas consultas:
- UFMG: "GLARMA Model for Temporal Data Analysis: ... a bootstrap proposal for inference on model parameters" (2024).
- UFMG: "Caracterizacao de um modelo de sinucleinopatia ..." (2024), com citacao explicita de Welch t test no resumo extraido.
- UFPE: "Aplicacao de Metodos Bootstrap na Construcao de Intervalos de Confianca para os parametros da Distribuicao Gama" (2022).
- UFF: "Concentracoes sericas de 25-hidroxivitamina D ..." (2025), com uso de teste t de Student ou Welch no resumo extraido.

Interpretacao:
- os metodos classicos (bootstrap, Welch, tamanho de efeito e erros robustos) continuam amplamente usados em pesquisas recentes;
- portanto, manter referencias fundacionais e metodologicamente adequado;
- complementar com evidencias recentes de aplicacao melhora a justificativa de atualidade.

Limitacoes de coleta:
- USP, UNICAMP e UFRGS nao tiveram extracao estruturada estavel via ferramenta automatica utilizada;
- UNIFESP retornou estrutura DSpace nova e exigiria refinamento de rotas de busca para extracao por termo;
- FGV redirecionou para portal institucional diferente, tambem exigindo busca manual assistida;
- UFABC aponta para catalogo bibliografico geral (nao um repositorio de teses no mesmo padrao DSpace), o que demanda estrategia de consulta distinta.

---

Escopo desta versao:
- Este documento descreve a metodologia estatistica implementada no codigo atual em `src/slm_benchmark/analysis.py` e sua integracao com a geracao de resultados em `scripts/analyze_results.py`.
