# **SPRINT 4 | Modelagem Linear para Aprendizado de Máquina **

### **Colaboradores**

*   Lucas Alves Antunes Almeida | RM: 566362
*   Lucas Werpp Franco | RM: 556044
*   Lucca Rosseto Rezende | RM: 564180
*   Massayoshi Bando Fogaça e Silva | RM: 561779
*   Miguel Lima da Silva | RM: 565141

 Projeto: Classificação de Eficiência Energética (Renewable Energy Dataset)
 Descrição do Projeto

Este projeto tem como objetivo analisar e classificar dados de produção de energia renovável (dataset: Renewable Energy Production Dataset 2010–2020
) utilizando técnicas de Machine Learning e modelos lineares de classificação.

O trabalho foi dividido em três partes principais:

Análise do Dataset – limpeza, encoding e correlação.

Definição e Treinamento do Modelo – aplicação de Regressão Logística (modelo linear).

Avaliação da Performance – comparação de métricas (acurácia, precisão e recall) em diferentes proporções de teste.

 Estrutura de Arquivos
 Projeto_Renewable_Energy
│
├── analise_dataset.py               # Parte 1 - Carregamento, encoding e correlação
├── regressao_logistica.py              # Parte 2 - Modelo linear (Regressão Logística)
├── avaliacao_da_performance.py         # Parte 3 - Métricas e comparação
│
├── renewable_energy_part1_treated.xlsx  # Base tratada gerada na Parte 1
├── Renewable_Energy_Data.csv            # Dataset original
│
└── README.md                        # Este arquivo (tutorial)

⚙️ Requisitos

Antes de executar os scripts, instale as dependências:

pip install pandas numpy scikit-learn seaborn matplotlib openpyxl

 Passo a Passo de Execução
 1. Análise do Dataset

Arquivo: analise_dataset.py

Funções:

Carrega o dataset CSV.

Faz o encoding da variável Energy_Class.

Gera a matriz de correlação entre as variáveis.

Salva o arquivo tratado em Excel (renewable_energy_part1_treated.xlsx).

Como executar:

python analise_dataset.py

 2. Definição do Modelo

Arquivo: definicao_modelo.py

Funções:

Carrega o arquivo tratado da Parte 1.

Define o modelo linear Logistic Regression (Scikit-Learn).

Realiza a classificação com 40% e 15% de dados de teste.

Como executar:

python definicao_modelo.py

 3. Avaliação da Performance

Arquivo: avaliacao_performance.py

Funções:

Gera as matrizes de confusão dos dois cenários.

Calcula acurácia, precisão e recall.

Exibe gráficos e comentários sobre as diferenças entre os testes de 40% e 15%.

Como executar:

python avaliacao_performance.py

 Conclusões

A Regressão Logística foi escolhida por ser um modelo linear, cuja fronteira de decisão é um hiperplano que separa as classes.

A matriz de correlação mostrou que as features possuem correlação moderada, justificando o uso de todas as variáveis numéricas no modelo.

O teste com 15% apresentou melhor acurácia (mais dados para treinar), mas o teste com 40% foi mais confiável (amostra maior para validação).

O equilíbrio ideal depende do tamanho do dataset e do objetivo da análise.

 Referências

Dataset: Renewable Energy Production 2010–2020 (Kaggle)

Artigo Alura: Problemas resolvidos com algoritmos de classificação

Alura: Machine Learning - Classificação por trás dos panos


