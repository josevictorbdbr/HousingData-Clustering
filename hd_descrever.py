import pickle
import pandas as pd

#abrir dados
dados = pd.read_csv('HousingData.csv')

#Separar dados dos imoveis
dados_imoveis = dados.drop(columns=['MEDV'])

#Carregar arquivos após treinamento
preenchedor_num = pickle.load(open('preenchedor_num_housing.pkl', 'rb'))
preenchedor_cat = pickle.load(open('preenchedor_cat_housing.pkl', 'rb'))
normalizador = pickle.load(open('normalizador_house.pkl', 'rb'))
cluster_housing = pickle.load(open('cluster_housing.pkl', 'rb'))
colunas_housing = pickle.load(open('colunas_housing.pkl', 'rb'))

#Separar Colunas
colunas_cat = ['CHAS']
colunas_num = [col for col in colunas_housing if col not in colunas_cat]

#Garantir mesma ordem colunas
dados_imoveis = dados_imoveis[colunas_housing]

#preencher valores que faltam
dados_num_preenchidos = preenchedor_num.transform(dados_imoveis[colunas_num])
dados_cat_preenchidos = preenchedor_cat.transform(dados_imoveis[colunas_cat])

dados_num_preenchidos = pd.DataFrame(dados_num_preenchidos, columns=colunas_num)
dados_cat_preenchidos = pd.DataFrame(dados_cat_preenchidos, columns=colunas_cat)

#Juntar para voltar as colunas ao original 
dados_imoveis_preenchidos = pd.concat([dados_num_preenchidos, dados_cat_preenchidos], axis=1)
dados_imoveis_preenchidos = dados_imoveis_preenchidos[colunas_housing]

#Normalizar
dados_imoveis_norm = normalizador.transform(dados_imoveis_preenchidos)
dados_imoveis_norm = pd.DataFrame(dados_imoveis_norm, columns=colunas_housing)
    
dados['cluster'] = cluster_housing.predict(dados_imoveis_norm)

#Mostrar quantidade de imoveis por cluster
print('\nQuantidade de imóveis por cluster:\n')
print(dados['cluster'].value_counts().sort_index().to_string())

#Mostrar media das variaveis por cluster
resumo_clusters = dados.groupby('cluster').mean(numeric_only=True)
print(resumo_clusters)