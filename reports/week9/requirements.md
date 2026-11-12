# Aula 9: requisitos e opção escolhida

Fonte principal: notebook local `aula 9/09_beyond_compas.ipynb`, em particular as células 0–8 e 12. A aula pede **escolher uma opção**; esta entrega segue a **opção A, Adult income**. O repositório público da professora, verificado em 12 de novembro de 2026, continua no [commit Week 8, 7dc7713](https://github.com/sofiacper/ETAI-Pipeline/tree/7dc7713640fe3b97f16be6df6be67ce417f9d950). Nesta aula, o código das adaptações deve ser feito no fork, e não copiado de uma nova versão da pipeline.

Prazo indicado no notebook: sábado às 23:59. O check semanal procura a opção escolhida a funcionar com configuração própria, as metas atingidas e a secção do README; na opção A também exige o CSV de previsões para as novas pessoas.

| Requisito da opção A | Implementação/evidência |
|---|---|
| Manter COMPAS a funcionar por omissão | `python main.py` continua a usar `config.yaml`; comparação antes/depois de dados, oito métricas nos cinco folds, probabilidades OOF e features finais em `compas_regression.json`. |
| Configuração própria | `config_adult.yaml`, escolhida pelo argumento `--config` que a pipeline já tinha. |
| Dados fornecidos no repositório | `data/adult_train.csv`, `data/adult_new.csv`, `data/ADULT_README.md`. |
| Corrigir dependências específicas do COMPAS | Limpeza de todas as categorias configuradas, conversão de missing para `np.nan`, auditoria com referência externa opcional, retorno imediato sem features de engenharia e inferência genérica. |
| Converter `?` em missing | Confirmação dos totais 379, 381 e 86 em workclass, occupation e native_country. |
| Não aplicar limites COMPAS ao Adult | Idades 17–90; capital gain/loss ≥0; horas 1–99. Preservados 74 casos com 17 anos e 29 ganhos top-coded de 99999. |
| Listar todos os atributos pretendidos | 23 numéricos e 12 categóricos, incluindo `v01`–`v25`, mais indicadores de workclass e occupation: 37 inputs transformados antes da seleção. |
| Identificadores, sampling weights e redundância | `id` não é preditor; `fnlwgt` e `education` são excluídos, mantendo `education_num`. |
| Escolha de atributo sensível | Auditar sex; excluir sex e race dos preditores. Justificação e limitação relativa a proxies no README. |
| Metas na mesma execução | ROC-AUC ≥0,90, accuracy ≥0,855 e F1 ≥0,66, nos scores externos de nested CV. Ver `target_checks.csv`. |
| Comparações com tuning e teste corrigido | Duas receitas XGBoost, seleção ligada/desligada, 30 trials, cinco folds externos/internos e teste emparelhado corrigido. |
| Prever os novos dados sem perder linhas | `predictions/adult_new_predictions.csv`: id, probabilidade e previsão 0/1; uma linha por cada uma das 2.000 entradas, na ordem original. |
| Outputs separados | Resultados e scoreboard Adult em `results/adult/`; evidência compacta em `reports/week9/`; scoreboard COMPAS preservado. |
| README | Linha W9, secção “Beyond COMPAS”, comandos, tabela de problemas/correções, resultados, features mantidas e reflexão de 3–5 frases. |

Os scores finais são os dos folds externos de nested CV, não o máximo otimista do estudo final. Os 1.200 registos de teste Adult continuam reservados e não são usados para escolher modelos ou ajustar a previsão final. Os ficheiros de modelos ajustados e probabilidades individuais OOF ficam localmente em `results/adult/`, seguindo o formato de entregas anteriores; o CSV de previsões pedido acompanha a entrega.

Resultados finais da receita configurada com seleção: **ROC-AUC 0,9092 ± 0,0162**, **accuracy 0,8571 ± 0,0109** e **F1 0,6657 ± 0,0288**, atingindo simultaneamente as três metas. Sem seleção: ROC-AUC 0,9073, accuracy 0,8600 e F1 0,6669. A diferença emparelhada de ROC-AUC é +0,001949, com p corrigido=0,1655; não se estabelece uma melhoria estatística. O seletor final mantém oito inputs e nenhum atributo `v`, após receber os 25 na pipeline. A opção com seleção fica configurada pela redução de inputs e cumprimento das metas; o README também mostra que o tuning não melhorou as médias dessa receita face aos parâmetros fixos.

Validação automática: 17 testes passaram, incluindo os testes anteriores COMPAS e os novos testes de missing counts, limites, inclusão de todos os `v`, exclusão de atributos protegidos, novas categorias e preservação de IDs/ordem/duplicados na inferência. A opção B (classificação multiclass do score COMPAS) é a alternativa à opção A e não é exigida juntamente com ela.
