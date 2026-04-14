import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist
import pickle
import math
import numpy as np

#Abrir o arquivo
dados = pd.read_csv('HousingData.csv')

#Separar valores e dados dos imoveis
dados_imoveis = dados.drop(columns=['MEDV'])
dados_valor = dados['MEDV']

#Separar colunas em categoricas ou numericas
colunas_cat = ['CHAS']
colunas_num = [col for col in dados_imoveis.columns if col not in colunas_cat]

#Tratar valores em todas as colunas
preenchedor_num = SimpleImputer(strategy='median')
preenchedor_cat = SimpleImputer(strategy='most_frequent')

#Prencher valores que faltam
dados_num_prenchidos = preenchedor_num.fit_transform(dados_imoveis[colunas_num])
dados_cat_prenchidos = preenchedor_cat.fit_transform(dados_imoveis[colunas_cat])

#Criar Data Frame de ambos dados prenchidos
dados_num_prenchidos = pd.DataFrame(dados_num_prenchidos, columns=colunas_num)
dados_cat_prenchidos = pd.DataFrame(dados_cat_prenchidos, columns=colunas_cat)

#Junta os dois
dados_imoveis_prenchidos = pd.concat([dados_num_prenchidos, dados_cat_prenchidos], axis=1)
dados_imoveis_prenchidos = dados_imoveis_prenchidos[dados_imoveis.columns]

#Salvar Preenchedores
pickle.dump(preenchedor_num, open('preenchedor_num_housing.pkl', 'wb'))
pickle.dump(preenchedor_cat, open('preenchedor_cat_housing.pkl', 'wb'))

#normalizar dados
normalizador = MinMaxScaler()
dados_imoveis_norm = normalizador.fit_transform(dados_imoveis_prenchidos)
dados_imoveis_norm = pd.DataFrame(dados_imoveis_norm, columns=dados_imoveis.columns)

print(dados_imoveis_norm)

#salvar normalizador
pickle.dump(normalizador, open('normalizador_house.pkl', 'wb'))

#distorcoes
distorcoes = []
K = range(1, 21) 

for i in K:
    modelo_clusters = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_imoveis_norm)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_imoveis_norm, modelo_clusters.cluster_centers_,'euclidean'), axis=1
                )/dados_imoveis_norm.shape[0]
            )
        )   
    
#Determinar numero otimo de clusters
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []

for i in range(len(distorcoes)):
    x= K[i]
    y= distorcoes[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distancias.append(numerador/denominador)
    
numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print('Numero otimo de clusters =', numero_clusters_otimo)

#Treinar modelo final
cluster_housing  = KMeans(n_clusters=numero_clusters_otimo, random_state=42, n_init=10).fit(dados_imoveis_norm)

#salvar modelo
pickle.dump(cluster_housing, open('cluster_housing.pkl', 'wb'))

#salvar nomes colunas
pickle.dump(list(dados_imoveis.columns), open('colunas_housing.pkl', 'wb'))

print('Treinamento concluido com sucesso')