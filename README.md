## Estrutura do Repositório

O repositório está organizado da seguinte forma:

- **artifacts**: Esta pasta contém os três melhores modelos e suas avaliações.

- **data**: Aqui, você encontrará os dados utilizados no desafio.

  - **raw**: Contém os dados originais enviados para o desafio.
  
  - **processado**: Inclui as bases de treino e teste após o pré-processamento.

- **training_utils**: Esta pasta contém funções e classes utilizadas no treinamento e avaliação dos modelos.

  - **assessment.py**: Um notebook de avaliação dos modelos.
  
  - **optuna_train.py**: Um notebook de treinamento dos modelos com o uso do Optuna.
  
  - **utils.py**: Arquivo com funções genéricas úteis.

- **eda.ipynb**: Um notebook Jupyter com a exploração inicial dos dados.

- **training.ipynb**: Um notebook Jupyter que contém o treinamento e a avaliação dos modelos.

- **pre_processing.ipynb**: Outro notebook Jupyter que descreve o pré-processamento, a criação de variáveis e a divisão dos dados.

- **utils.py**: Um arquivo Python com funções genéricas úteis.
