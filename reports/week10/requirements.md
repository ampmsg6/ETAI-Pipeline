# Aula 10: requisitos e evidência

Fontes principais: `aula 10/10_missing_values.ipynb`, células 11–15 e 24, e `aula 10/11_deployment.ipynb`, células 14–18 e 23. A referência pública é o [commit e398cb8 — Week 10: Missing Values & Deployment](https://github.com/sofiacper/ETAI-Pipeline/tree/e398cb88afa4e92bc7d0e8b42443f95ebcb570ca), publicado pela professora em 17 de novembro de 2026 e verificado em 20 de novembro. A entrega semanal exige a Parte A e a tarefa do README até sábado às 23:59.

| Requisito | Implementação e evidência |
|---|---|
| A1: novos imports e `_NUMERIC_IMPUTERS` | `src/preprocessing.py`: median, mean, KNN, IterativeImputer e IterativeImputer com RandomForestRegressor. Mantida a opção constant para scripts antigos. |
| A1: escalar antes de imputar | Pipeline numérica `scale → impute`, aprendida só no treino de cada fold; teste de distâncias KNN e estatísticas de treino. |
| A1: opções de configuração | `preprocessing.imputation.numeric_strategy`, `n_neighbors`, `max_iter`, `n_estimators` e seed. Mantidos os planos individuais de indicadores e os overrides usados pelos scripts anteriores. |
| A1: quatro execuções | `python -m scripts.compare_week10`: median, knn, iterative e random_forest; XGBoost fixo e seleção ligada, tuning desligado, mesmos dados, seeds e cinco folds. |
| A1: teste corrigido | `imputer_comparisons.csv`: seis testes emparelhados sobre ROC-AUC, rácio validação/treino 1/4; p sem ajuste para múltiplas comparações. |
| Registar o imputer | Console, scoreboard, cabeçalho do relatório e configuração de cada execução. |
| A2: guardar a pipeline completa | `main.py` chama `save_model()` depois do refit no desenvolvimento, desativa a cache e guarda `output.model_path`; modelos locais, excluídos de Git. |
| A2: nova previsão pelo terminal | `predict.py`: usa o modelo e a configuração guardados, sem treinar, limpar duplicados ou consultar labels. |
| A2: dados e 20 respostas | `data/compas_new_people.csv`, copiado do material local; comando exigido escreve 20 linhas em `predictions.csv`. Cópia para revisão em `predictions/compas_new_predictions.csv`. |
| A2: dependências e exclusões | `joblib` declarado em `requirements.txt`; `models/` e `predictions.csv` no `.gitignore`. |
| README | Linha W10, quatro execuções, teste corrigido, escolha de imputer em três frases e comando/conteúdo das previsões. |
| Extra B4: prever com validação | Erro claro para colunas em falta; versões, configuração e checksum junto do modelo; teste de serialização, ausência de refit e preservação de IDs/ordem/linhas repetidas. |
| Compatibilidade Adult | Configuração com modelo separado; execução sem tuning e 2.000 previsões em paths de validação, preservando a entrega da aula 9. |

Resultados de CV: median **0,7208 ± 0,0125**, KNN **0,7193 ± 0,0104**, iterative **0,7214 ± 0,0136**, random_forest **0,7224 ± 0,0125** de ROC-AUC. Nenhum dos seis testes corrigidos tem p<0,05; conserva-se a mediana pela simplicidade. A maior média observada pertence à floresta, mas o ganho de 0,0016 face à mediana tem p=0,4253. Estes resultados não provam equivalência.

As quatro receitas usam 5.771 registos de desenvolvimento, os mesmos cinco folds e XGBoost com 300 árvores, learning rate 0,05, profundidade 4 e seed 42; `VoteSelector` usa dois votos. Os 1.443 registos reservados ficam fora do treino final e da avaliação. As médias são de CV simples, sem tuning, conforme permitido no enunciado. A floresta do imputer tem 50 árvores por coluna e no máximo dez iterações; em alguns ajustes atingiu esse limite sem satisfazer a tolerância de paragem. Os tempos incluem toda a execução e não constituem um benchmark isolado de imputação.

Adaptações ao exemplo da professora: preservação da interface de configuração completa deste projeto, dos indicadores e da opção Adult; um worker na floresta porque os folds já correm em paralelo; preservação de colunas completamente vazias em treino; configuração de treino guardada junto do modelo para a inferência continuar a usar as mesmas regras. A mudança para `scale → impute` é intencional: os resultados históricos das semanas 8–9 permanecem identificados como históricos, com outra chave de comparação.

Evidência: `scoreboard.csv`, `folds.csv`, `metrics.csv`, `features.csv`, `imputer_comparisons.csv`, relatórios/configurações, `metadata.json` e `validation.json`. Os 22 testes automáticos passaram. A verificação independente recalcula as oito métricas nos 20 folds a partir das probabilidades OOF, recalcula os seis testes corrigidos, verifica ausência de registos reservados nas OOF, executa o CLI real e confirma as 20 previsões e o erro para colunas em falta. Modelos e probabilidades individuais permanecem localmente em `models/` e `results/`, seguindo o formato das entregas anteriores.

B1 (imputação múltipla), B2 (dar categorias ao imputer) e B3 (proximidades de florestas) são extras e não integram esta entrega.
