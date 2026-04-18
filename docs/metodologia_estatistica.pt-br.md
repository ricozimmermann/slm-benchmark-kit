# Metodologia Estatística do SLM Benchmark Kit

## 0. Justificativa científica: por que SLMs e por que benchmark

O interesse por Small Language Models (SLMs) é consistente com uma mudança de foco na IA aplicada: sair de soluções exclusivamente baseadas em escala máxima e avançar para modelos eficientes, executáveis em cenários com restrições reais de custo, energia, latência e privacidade.

No contexto prático, SLMs são particularmente relevantes porque:
- reduzem requisitos de memória e processamento, viabilizando uso em edge/mobile e em ambientes com infraestrutura limitada;
- permitem processamento local/offline, com menor dependência de nuvem e menor exposição de dados sensíveis;
- favorecem a democratização tecnológica, ampliando o acesso para instituições com menor capacidade computacional;
- podem entregar desempenho competitivo em domínios específicos quando bem ajustados e avaliados.

Entretanto, a maior eficiência de um SLM não implica automaticamente melhor desempenho útil. Por isso, a comparação entre modelos precisa ser feita sob protocolo experimental controlado, com inferência estatística explícita e métricas operacionais complementares.

Assim, a justificativa para o benchmark neste projeto é dupla:
1. metodológica: medir diferenças de desempenho com controle de incerteza e de confundimento;
2. aplicada: identificar o melhor equilíbrio entre qualidade, estabilidade, latência e custo computacional para cenários reais.

Em síntese, SLMs são estratégicos por eficiência e acessibilidade; benchmark reprodutível é estratégico por confiabilidade da decisão técnica.

## 1. Resumo executivo

Este documento formaliza o conteúdo estatístico utilizado no projeto `slm-benchmark-kit` para comparar SLMs com foco em:
- estimativa com incerteza explícita;
- comparação inferencial entre modelos;
- controle de covariáveis experimentais;
- diagnóstico de qualidade do sistema de avaliação automática (judges).

A pipeline atual combina:
- estatística descritiva por modelo;
- intervalo de confiança bootstrap para a média de score;
- teste t de Welch para comparação entre dois modelos;
- tamanho de efeito de Cohen (d);
- regressão OLS com erros robustos (cluster por item quando disponível; HC3 como fallback);
- métricas de concordância e saúde dos juízes.

## 2. Unidade de análise e desenho experimental

## 2.1 Unidade observacional

A unidade observacional é o `trial` registrado no JSONL de resultados, contendo:
- identificação do modelo (`model`);
- identificação do item (`item_id`), tipo de tarefa e dificuldade;
- hiperparâmetros de geração (`temperature`, `top_p`, `top_k`);
- repetição (`repetition`);
- score agregado (`score_aggregated`) quando há pelo menos um juiz válido.

## 2.2 Definições operacionais

- `valid_response`: no contexto da pipeline atual, significa que pelo menos um juiz retornou score válido (`judge_valid_count > 0`).
- `score_aggregated`: mediana dos scores válidos dos juízes para o trial.
- `error`: erro de geração (por exemplo, timeout). Quando presente, o trial tende a ficar sem score agregado.

## 2.3 Estratégias para reduzir vieses

O desenho implementado no benchmark contempla:
- seed fixa para reprodutibilidade;
- randomização da ordem de execução dos trials;
- repetições por combinação de parâmetros;
- separação por split de avaliação (`eval_split`) para evitar leakage entre tuning e teste;
- avaliação por múltiplos juízes com agregação robusta por mediana.

## 3. Variáveis e métricas reportadas

## 3.1 Performance principal

Para cada modelo, são reportados:
- `n`: total de trials;
- `n_scored`: quantidade de trials com `score_aggregated` numérico;
- média, desvio-padrão amostral e mediana de `score_aggregated`;
- intervalo de confiança bootstrap de 95% da média (`ci95_low`, `ci95_high`).

## 3.2 Confiabilidade operacional

Também são reportados:
- `valid_rate` (média de `valid_response`);
- `error_rate` (proporção de erros não vazios);
- `timeout_rate` (proporção de erros contendo "timeout");
- `judge_all_failed_rate` (proporção de trials com `judge_valid_count <= 0`);
- `latency_p50_ms` e `latency_p95_ms`.

Essas métricas evitam conclusões baseadas apenas em score médio e evidenciam custo/estabilidade.

## 4. Métodos de inferência estatística

## 4.1 Intervalo de confiança bootstrap para a média

Para cada modelo, o IC de 95% da média é estimado por bootstrap percentile:
1. amostrar com reposição, `n_boot = 5000`, do vetor de scores válidos;
2. calcular a média em cada reamostragem;
3. usar os percentis 2.5% e 97.5% da distribuição bootstrap.

Configuração atual:
- `seed = 42` no bootstrap.

Interpretação:
- IC mais estreito indica maior precisão da estimativa média;
- sobreposição de ICs não substitui teste formal de hipótese.

## 4.2 Teste t de Welch (comparação entre dois modelos)

Quando há exatamente dois modelos com ao menos 2 observações válidas por grupo, aplica-se o teste t de Welch (variâncias desiguais):

- Hipótese nula: médias populacionais iguais.
- Hipótese alternativa: médias populacionais diferentes.

O relatório inclui:
- `t_stat`;
- `p_value`;
- indicador `significant_0_05`.

Observação de escopo:
- a implementação atual executa Welch apenas no caso de 2 modelos; para mais de 2 modelos, o teste é pulado com aviso.

## 4.3 Tamanho de efeito (Cohen's d)

Junto ao Welch, é calculado Cohen's d para quantificar a magnitude do efeito:

- d ~ 0.2: efeito pequeno (regra prática);
- d ~ 0.5: efeito médio;
- d ~ 0.8: efeito grande.

Boas práticas:
- interpretar `p_value` em conjunto com `d`;
- reportar também a diferença de médias (`mean_a - mean_b`) e o contexto aplicado.

## 4.4 Regressão OLS com erros robustos

Para controlar confundimento de configuração e composição de tarefas, o projeto ajusta:

`score_aggregated ~ C(model) + temperature + top_p + top_k + C(task_type)`

Detalhes:
- `model` e `task_type` entram como fatores categóricos;
- o ajuste ocorre somente se houver colunas obrigatórias e ao menos 8 observações válidas;
- se `item_id` possuir mais de um valor único, usa-se covariância cluster-robust por item;
- em caso contrário (ou falha), usa-se HC3.

Interpretação recomendada:
- coeficientes de `C(model)` representam diferenças condicionais (controladas por covariáveis);
- erros robustos reduzem sensibilidade à heterocedasticidade;
- cluster por item ajuda a lidar com dependência intra-item entre repetições.

## 5. Qualidade do sistema de avaliação automática (judges)

## 5.1 Concordância par a par

A partir de `judge_scores`, o sistema calcula por par de juízes:
- Spearman;
- Kendall tau;
- Pearson;
- MAE (erro absoluto médio);
- `n_overlap` (amostras com score válido em ambos).

Tratamento de casos-limite:
- se `n_overlap < 3`, as correlações são reportadas como `NaN`;
- se uma série for constante, as correlações são `NaN` e o MAE permanece calculado.

## 5.2 Saúde dos juízes

A partir de `judge_rationales`, o sistema estima por juiz:
- `valid_rate`;
- `parse_error_rate`;
- `judge_error_rate`;
- `parse_fallback_rate`.

Essas métricas permitem diferenciar:
- divergência de critério (baixa concordância);
- falha técnica de parse/infraestrutura (baixa saúde operacional).

## 6. Ameaças à validade e limitações

## 6.1 Validade interna

- Dependência entre trials: repetições do mesmo `item_id` não são independentes em sentido estrito (mitigado parcialmente com cluster-robust no OLS).
- Seleção por respostas válidas: análises de score usam `score_aggregated` não nulo; diferenças de taxa de falha entre modelos podem enviesar comparações de média.

## 6.2 Validade externa

- Resultados dependem do dataset atual (`slm_tasks_ptbr.jsonl`), idioma e distribuição de tarefas.
- Generalização para outros domínios requer replicação com novos conjuntos estratificados.

## 6.3 Validade de construto

- O score agregado depende da rubrica dos juízes (heurística/SLM ou LLM) e da robustez do parse.
- Concordância alta não implica validade semântica total; por isso recomenda-se avaliação humana cega complementar.

## 7. Reprodutibilidade e reporte científico

Para reporte auditável, recomenda-se sempre publicar:
- JSONL bruto;
- relatório markdown gerado pela pipeline;
- YAML de configuração exato;
- hash de commit;
- versão do dataset e checksum;
- versão do protocolo (`VERSION` + `CHANGELOG.md`).

Checklist operacional complementar:
1. executar benchmark com `eval_split: test` para conclusões finais;
2. registrar falhas (`error_rate`, `timeout_rate`) junto aos scores;
3. reportar efeito + incerteza (diferença de médias, IC, `p_value`, `d`);
4. evitar conclusões com base exclusiva em limiar de 0.05.

## 8. Diretrizes de interpretação

Ao comparar modelos, priorize esta ordem:
1. viabilidade operacional (valid_rate, error_rate, timeout_rate, latência);
2. estimativa central e incerteza (média + IC bootstrap);
3. evidência inferencial (Welch + tamanho de efeito);
4. análise ajustada (OLS robusto) para robustez da conclusão.

Um modelo com média maior, mas alta taxa de falha, pode ser pior em um cenário real do que um modelo com média ligeiramente menor e estabilidade superior.

## 9. Referências metodológicas (ABNT)

- CAMERON, A. Colin; MILLER, Douglas L. A Practitioner's Guide to Cluster-Robust Inference. Journal of Human Resources, v. 50, n. 2, p. 317-372, 2015. DOI: https://doi.org/10.3368/jhr.50.2.317.
- COHEN, Jacob. Statistical Power Analysis for the Behavioral Sciences. 2. ed. Hillsdale, NJ: Lawrence Erlbaum Associates, 1988. ISBN: 9780805802832.
- EFRON, Bradley; TIBSHIRANI, Robert J. An Introduction to the Bootstrap. New York: Chapman & Hall, 1993. ISBN: 9780412042317.
- MACKINNON, James G.; WHITE, Halbert. Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties. Journal of Econometrics, v. 29, n. 3, p. 305-325, 1985. DOI: https://doi.org/10.1016/0304-4076(85)90158-7.
- WELCH, B. L. The Generalization of Student's Problem when Several Different Population Variances are Involved. Biometrika, v. 34, n. 1-2, p. 28-35, 1947. DOI: https://doi.org/10.2307/2332510.

### 9.1 Nota de rastreabilidade

As referências desta seção foram verificadas por metadados bibliográficos em fontes externas:
- artigos: Crossref (título, autores, periódico, volume, número, páginas e DOI);
- livros: catálogos ISBN (Open Library/Google Books) para autores, edição, editora e ano.

Data da verificação: 2026-04-15.

### 9.2 Evidência de uso em repositórios brasileiros

Para responder a uma possível dúvida sobre "atualidade" dos métodos, foi realizada consulta adicional automatizada em repositórios institucionais brasileiros. O objetivo não foi substituir as referências fundacionais, mas verificar uso recente em teses e dissertações.

Resumo dos retornos observados:
- UFMG ([repositorio.ufmg.br](https://repositorio.ufmg.br/)): busca por bootstrap com 580 resultados; busca por welch com 394 resultados.
- UFPE ([repositorio.ufpe.br](https://repositorio.ufpe.br/)): busca por bootstrap com 1196 resultados; busca por welch com 766 resultados.
- UFF ([app.uff.br/riuff](https://app.uff.br/riuff/)): busca por bootstrap com 34 resultados; busca por welch com 35 resultados.

Exemplos de itens retornados nas consultas:
- UFMG: "GLARMA Model for Temporal Data Analysis: ... a bootstrap proposal for inference on model parameters" (2024).
- UFMG: "Caracterização de um modelo de sinucleinopatia ..." (2024), com citação explícita de Welch t test no resumo extraído.
- UFPE: "Aplicação de Métodos Bootstrap na Construção de Intervalos de Confiança para os parâmetros da Distribuição Gama" (2022).
- UFF: "Concentrações séricas de 25-hidroxivitamina D ..." (2025), com uso de teste t de Student ou Welch no resumo extraído.

Interpretação:
- os métodos clássicos (bootstrap, Welch, tamanho de efeito e erros robustos) continuam amplamente usados em pesquisas recentes;
- portanto, manter referências fundacionais é metodologicamente adequado;
- complementar com evidências recentes de aplicação melhora a justificativa de atualidade.

Limitações de coleta:
- USP, UNICAMP e UFRGS não tiveram extração estruturada estável via ferramenta automática utilizada;
- UNIFESP retornou estrutura DSpace nova e exigiria refinamento de rotas de busca para extração por termo;
- FGV redirecionou para portal institucional diferente, também exigindo busca manual assistida;
- UFABC aponta para catálogo bibliográfico geral (não um repositório de teses no mesmo padrão DSpace), o que demanda estratégia de consulta distinta.

---

Escopo desta versão:
- Este documento descreve a metodologia estatística implementada no código atual em `src/slm_benchmark/analysis.py` e sua integração com a geração de resultados em `scripts/analyze_results.py`.