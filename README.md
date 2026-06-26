#  Previsão de Potabilidade da Água: Uma Abordagem de Machine Learning para Saúde Pública

> ** Apresentação:** [Clique aqui para acessar os Slides da Apresentação do Trabalho de Conclusão](#adicione_o_seu_link_aqui)

##  Contexto do Projeto
Garantir o acesso à água potável é um dos maiores desafios da saúde pública global. Este projeto tem como objetivo desenvolver um modelo preditivo capaz de classificar amostras de água como potáveis (Classe 1) ou não potáveis (Classe 0) com base em variáveis químicas (pH, Sulfato, Dureza, Sólidos, etc.). 

A base de dados utilizada (Kaggle Water Potability) apresenta um alto grau de complexidade, ruído e sobreposição matemática entre as classes, exigindo técnicas avançadas de pré-processamento e otimização de hiperparâmetros para evitar riscos sanitários (Falsos Positivos).

---

##  Metodologia e Pipeline de Dados

1. **Pré-processamento Avançado:** O dataset original possuía taxa relativa de dados ausentes em variáveis vitais, como Sulfato (23,8%) e pH (14,9%). Para não distorcer a distribuição, evitou-se a imputação por média. Optou-se pela padronização via `StandardScaler` seguida de imputação multidimensional via `KNNImputer` (K=5).
2. **Engenharia de Features:** Criação de variáveis categóricas baseadas nas diretrizes da **Organização Mundial da Saúde (OMS)**, fornecendo aos algoritmos limites biológicos reais (ex: pH seguro entre 6.5 e 8.5).
3. **Avaliação Focada em Saúde:** A métrica principal não foi a Acurácia (que é ilusória em dados desbalanceados), mas sim o **F1-Score da Classe 1** e a minimização de Falsos Positivos, balanceando a detecção de água boa com o rigor sanitário.

---

##  Resultados e Comparação de Modelos

Para comprovar a complexidade não-linear do problema e encontrar o ponto de equilíbrio ideal, testamos e otimizamos rigorosamente quatro abordagens matemáticas diferentes:

| Modelo | Acurácia | F1-Score (Água Potável) | Falsos Positivos | Comportamento / Veredito |
| :--- | :---: | :---: | :---: | :--- |
| **Regressão Logística (LR)** | 61% | 0.00 | 0 | **Baseline (Falha Proposital):** Provou que as variáveis não são linearmente separáveis. A penalidade Lasso zerou os coeficientes, chutando tudo como Classe 0. |
| **Random Forest (RF)** | 65% | 0.48 | 84 | **Robustez:** Equilíbrio clássico de florestas aleatórias. |
| **Extra Trees (ETC)** | **67%** | 0.49 | **64** | **O Mais Seguro:** Realizou uma "poda" agressiva do ruído. Tem a maior precisão, sendo excelente para evitar que pessoas fiquem doentes, mas sacrifica muita água boa. |
| **Support Vector Machine (SVM)** | 62% | **0.51** | 119 | **O Agressivo:** Usando limites geométricos (`C=10`, `RBF`), encontrou o maior volume de amostras de água potável, mas com maior risco sanitário. |
| **XGBoost (Campeão)** | 64% | **0.51** | 102 | **O Ponto de Equilíbrio:** Aliado às regras de negócio da OMS, obteve o melhor balanço geral. Maximizou o acerto na classe minoritária sem explodir os falsos positivos. |

###  Conclusão Científica
A estagnação das métricas de acurácia global na faixa de 64% a 67% **não representa uma falha de modelagem, mas sim o limite teórico de previsibilidade do dataset**. A análise de densidade (KDE) comprovou que amostras quimicamente idênticas possuem rótulos opostos. Os modelos otimizados neste projeto conseguiram extrair o máximo de informação útil, estabelecendo um *trade-off* perfeito entre segurança pública e aproveitamento hídrico.

---

##  Como Executar o Projeto

1. Clone este repositório:
   ```bash
   git clone [[https://github.com/SEU_USUARIO/trabalho-final-tc.git](https://github.com/Francionn/trabalho-topicos-computa-o.git)]([https://github.com/SEU_USUARIO/trabalho-final-tc.git](https://github.com/Francionn/trabalho-topicos-computa-o.git))
