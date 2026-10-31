# Aula 7: requisitos para a entrega

Extraído dos notebooks locais `aula 7/06_boosting.ipynb` e `aula 7/07_metrics.ipynb`, sobretudo das células 69–78 de `07_metrics`. Implementação consultada: [repositório da professora, commit a170b26](https://github.com/sofiacper/ETAI-Pipeline/tree/a170b26b131659ec9b5aff6d818d21354c3930c2).

O notebook indica entrega no sábado às 23:59. A Parte A e a tarefa do README são obrigatórias; o `VoteSelector` da Parte B é um desafio extra.

| Requisito obrigatório | Alteração neste projeto | Estado |
|---|---|---|
| A1: novos modelos | `gradient_boosting`, `hist_gradient_boosting` e `naive_bayes` em `src/model.py`, disponíveis no CLI | Implementado |
| A1: configuração e tuning | `model.type: gradient_boosting`; espaços de procura dos três modelos em `config.yaml` | Implementado |
| A2: oito métricas | Accuracy, balanced accuracy, precision, recall, F1, F2, ROC-AUC e average precision | Implementado |
| A2: métrica de seleção | `cv.selection_metric: roc_auc`, usada em Optuna e na ordenação da tabela | Implementado |
| A2: probabilidades OOF | CV normal e nested devolvem probabilidades; classificação e fairness usam limiar 0,5 | Implementado |
| A2: relatórios | Média/desvio de todas as métricas, folds, matriz de confusão, fairness e otimismo do tuning | Implementado |
| A3: scoreboard | `save_scores()`, baseline dummy nos mesmos folds e `output.scoreboard: results/runs.csv` | Implementado |
| A3: resultados comparáveis | Identificação dos dados, limpeza, folds e software; uma linha adicionada por execução | Implementado |
| A4: executar os nove modelos | Dummy, logística, Naive Bayes, árvore, RF, RF oblíqua, AdaBoost, GB e HistGB; tuning onde existe espaço | Concluído: nove modelos; 30 tentativas por estudo nos oito modelos com tuning |
| README | Linha W7, justificação da métrica, tabela com os nove modelos e instruções de reprodução | Concluído |

As regras próprias de limpeza e imputação, a exclusão de `race` e dos scores COMPAS dos preditores e a separação entre desenvolvimento e teste foram mantidas. A aula não exige novas dependências. A seleção de features por votação, os exemplos de regressão e as demonstrações manuais de boosting são material de aprendizagem ou extras, sem integração obrigatória nesta pipeline de classificação.

Validação automática: `python -m unittest discover -s tests -v` — 10 testes passaram, incluindo isolamento do preprocessing nos folds, probabilidades OOF, métricas, modelos novos, scoreboard, fairness com score COMPAS desconhecido e execução completa sem tuning.

Execuções finais: `python -m scripts.compare_week7 --trials 30 --resume`. Cada modelo com tuning usa cinco folds externos, cinco internos e 30 tentativas por estudo, mais um estudo final para refit. O dummy não tem tuning. Os 1.443 registos de teste ficam reservados.

Os resultados finais estão em [scoreboard.csv](scoreboard.csv), [metrics.csv](metrics.csv) e [folds.csv](folds.csv). Gradient Boosting obteve a maior média de ROC-AUC (0,7284 ± 0,0153); AdaBoost, HistGradientBoosting e RF oblíqua ficaram muito próximos, sem diferença estabelecida pelos testes emparelhados corrigidos entre esses quatro modelos.

Ficheiros para a entrega: `README.md`, `config.yaml`, código atualizado, `results/runs.csv` e os relatórios compactos desta pasta. As probabilidades individuais e todos os trials ficam localmente em `results/week7/`.
