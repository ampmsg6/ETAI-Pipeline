# Aula 8: requisitos da entrega

Fontes: notebook local `aula 8/08_feature_selection.ipynb`, células 52–60, e [projeto da professora, commit 7dc7713 — Week 8](https://github.com/sofiacper/ETAI-Pipeline/tree/7dc7713640fe3b97f16be6df6be67ce417f9d950), publicado em 2 de novembro de 2026. O notebook exige a Parte A e a tarefa do README até sábado às 23:59; os desafios B1 e B2 são extras.

| Requisito | Implementação nesta entrega |
|---|---|
| A1: instalar e declarar `xgboost>=2.0` | `requirements.txt` atualizado; versão instalada em `requirements-lock.txt`. |
| A1: registar `XGBClassifier` e configurar XGBoost | Registo em `src/model.py`, opção no CLI e `model.type: xgboost` em `config.yaml`. |
| A1: espaço de tuning | Nove parâmetros da professora, incluindo `gamma`, `reg_alpha`, `reg_lambda` e `min_child_weight`. |
| A1: executar sem tuning | Teste integrado de XGBoost sem tuning, com relatórios, scoreboard e features selecionadas. |
| A2: `VoteSelector` | RF com 200 árvores e `min_samples_leaf=20`, Gradient Boosting e logística L1 com `C=0.01`; mínimo de 1–3 votos. |
| A2: seleção dentro da pipeline | Ordem `prep → select → model`, preservando a interface de configuração completa deste projeto. Aprende apenas nos dados de treino de cada fold. |
| A2: configuração, cache e tuning conjunto | Secção `feature_selection`; cache temporária por execução; `select__min_votes` combinado com o espaço do modelo. A cache é removida também em caso de erro. |
| A2: mostrar as features finais | Mensagem `Features kept: ...`, votos por feature, número final de votos e coluna `feature_selection` no scoreboard e cabeçalho do relatório. |
| A3: duas execuções com tuning | `python -m scripts.compare_week8 --trials 30 --resume`, com seleção ligada e desligada, mesmos dados, seeds, orçamento e folds. |
| A3: teste emparelhado corrigido | `corrected_paired_test()` aplicado aos cinco ROC-AUC externos; diferenças calculadas como seleção ligada menos desligada, rácio validação/treino de 1/4. |
| README | Linha W8, tabela das duas execuções, teste corrigido, features mantidas e conclusão de 3–5 frases sobre a seleção. |

Cada receita usa 30 tentativas por estudo, cinco folds externos e cinco internos, mais um estudo final de 30 tentativas em todos os dados de desenvolvimento para escolher os parâmetros do refit. Os scores reportados para a comparação são os de nested CV; o melhor score do estudo final continua identificado como score de seleção. Os 1.443 registos de teste ficam reservados.

O [scoreboard](scoreboard.csv), os [folds](folds.csv), as [métricas](metrics.csv), as [features e votos finais](features.csv), os [trials dos estudos finais](trials.csv), o [teste corrigido](selection_comparison.csv) e os relatórios completos desta pasta documentam os resultados. Os metadados incluem o hash dos dados, versões, seeds e orçamento. As probabilidades OOF individuais ficam localmente em `results/week8/`, como nas entregas anteriores; `results/runs.csv` acompanha a entrega.

Resultados finais: ROC-AUC de nested CV **0,7260 ± 0,0129 com seleção** e **0,7285 ± 0,0146 sem seleção**; diferença emparelhada de −0,0025, teste corrigido com p=0,3286. O modelo final com seleção mantém `age` e `priors_count`, duas das dez features transformadas, com `min_votes=2`. A configuração conserva a seleção ligada pela redução de inputs e proximidade dos scores observados; o README distingue ausência de diferença detetada de prova de equivalência. Os cinco estudos externos escolheram mínimos de votos 1, 1, 3, 1 e 2.

A implementação mantém a limpeza e imputação já justificadas, a exclusão de `race`, `id` e dos scores COMPAS dos preditores, e a avaliação por oito métricas. As adaptações ao exemplo da professora limitam a floresta do seletor a um worker por fold, fixam também a seed da logística e validam o contrato de estimador; os três modelos, regras de voto e espaços de procura são os mesmos. Não se usa early stopping com as labels de validação ou teste.

Validação: `python -m unittest discover -s tests -v` — 13 testes passaram, incluindo seleção ajustada apenas nos folds internos de treino, isolamento dos folds externos, reutilização da cache em trials repetidos, nomes das features, compatibilidade com clonagem, métricas/OOF e execução de XGBoost sem tuning. Os testes e scripts das semanas 6–7 desativam explicitamente a nova seleção para conservar as suas receitas originais.

B1 (comparação com outros seletores) não integra a entrega obrigatória. B2 está documentado nos folds externos através de `select__min_votes`, do número de features e dos seus nomes; a interpretação encontra-se no README. Os exemplos didáticos de árvores para multiclass e regressão e o formulário de feedback não são alterações exigidas à pipeline.
