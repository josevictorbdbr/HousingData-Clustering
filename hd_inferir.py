import pickle
import pandas as pd

#Carregar arquivos salvos
preenchedor_num = pickle.load(open('preenchedor_num_housing.pkl', 'rb'))
preenchedor_cat = pickle.load(open('preenchedor_cat_housing.pkl', 'rb'))
normalizador = pickle.load(open('normalizador_house.pkl', 'rb'))
cluster_housing = pickle.load(open('cluster_housing.pkl', 'rb'))
colunas_housing = pickle.load(open('colunas_housing.pkl', 'rb'))

#Separar colunas
colunas_cat = ['CHAS']
colunas_num = [col for col in colunas_housing if col not in colunas_cat]

#Criar novo imovel
novo_imovel = pd.DataFrame([[
    0.10,   #CRIM
    0.0,    #ZN
    7.50,   #INDUS
    0.0,    #CHAS
    0.50,   #NOX
    6.20,   #RM
    65.0,   #AGE
    4.00,   #DIS
    5.0,    #RAD
    300.0,  #TAX
    17.0,   #PTRATIO
    390.0,  #B
    10.0    #LSTAT
]], columns=colunas_housing)

#Preencher valores que faltam
num_preenchido = preenchedor_num.transform(novo_imovel[colunas_num])
cat_preenchido = preenchedor_cat.transform(novo_imovel[colunas_cat])

num_preenchido = pd.DataFrame(num_preenchido, columns=colunas_num)
cat_preenchido = pd.DataFrame(cat_preenchido, columns=colunas_cat)

novo_imovel_preenchido = pd.concat([num_preenchido, cat_preenchido], axis=1)
novo_imovel_preenchido = novo_imovel_preenchido[colunas_housing]

#Normalizar
novo_imovel_norm = normalizador.transform(novo_imovel_preenchido)
novo_imovel_norm = pd.DataFrame(novo_imovel_norm, columns=colunas_housing)

#Cluster do novo imovel
cluster_novo_imovel = cluster_housing.predict(novo_imovel_norm)

print('Cluster do novo imovel:', cluster_novo_imovel[0])