# HousingData Clustering

Este projeto usa o algoritmo K-Means para agrupar imóveis em diferentes grupos com base nas suas características.

**Dataset:** HousingData.csv 

https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset

## Como Rodar

1. Dentro da pasta, criar e ativar o ambiente virtual
2. Instalar as dependências dentro do requirements.txt
3. Rodar o arquivo HousingDataTreino.py para ler o dataset, normalizar os dados, treinar o modelo de clusters e salvar os arquivos .pkl

## Módulos

- **HousingDataDescrever.py**  Aplica o modelo em todo o dataset e mostra quantos imóveis há em cada cluster
- **HousingDataInferir.py**  Cria um imóvel manualmente, aplica o mesmo pipeline e informa a qual cluster ele pertence
