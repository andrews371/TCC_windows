# -*- coding: utf-8 -*-
"""
Created on Tue June 02

@author: André
"""

import pandas as pd
base = pd.read_csv('credit_data.csv')
base.describe()

# 1) VALORES INCONSISTENTES
# valores de idade < 0 são inconsistentes
base.loc[base['age'] < 0] 
# tbm poderia fazer assim:
base.loc[base['age'] < 0, 'age']
# tbm poderia fazer essa busca assim:
base.loc[base.age < 0]
# ou assim:
base.loc[base.age < 0, 'age']

# FORMAS NÃO TÃO APROPRIADAS DE SE CORRIGIR

# 1. apagar a coluna inteira
base.drop('age', 1, inplace=True) # o 1 é para apagar a coluna inteira
# inplace=True é para rodar e jogar o resultado na própria variável "base" e não
# em outra que receberia "base"

# esse comando altera o dataframe (variável) "base" e não o arquivo .csv original

# 2. apagar somente as linhas com problema. muitas vezes é a melhor escolha
base.drop(base[base.age < 0].index, inplace=True)

# 3. preencher os valores manualmente (se for viável e não muito trabalhoso
# é a melhor escolha)

# BOA RESOLUÇÃO
# 4. preencher os valores inconsistentes com a média dos valores daquele atributo
# média de todos os atributos
base.mean() 

# média apenas da idade. aqui leva em conta os valores inconsistentes
base['age'].mean() 

# o jeito certo de pegar a média é não considerando os valores inconsistentes
# como segue abaixo
base['age'][base.age > 0].mean()

# atribuindo à media das idades válidas para os 3 valores de idade inconsistentes
base.loc[base.age < 0, 'age'] = 40.92


# 2) TRATAMENTO DE VALORES FALTANTES
# veificando valores nulos
pd.isnull(base)

# conta quantos atributos nulos existe em cada coluna
pd.isnull(base).sum()

# detectada a(s) coluna(s) que contém valores nulos, passamos à próxima etapa
# aqui diz se é verdadeiro ou falso que uma linha tem o atributo 'age' nulo
pd.isnull(base['age']) # pd.isnull(base.age)

# aqui é mais específico pois dis exatamente quais as linhas nulas do atributo
# em questão 'age'
base.loc[pd.isnull(base['age'])]

# TRATAMENTO

# antes vamos estar dividindo a base em previsores e classe

# o método .iloc faz essa divisão, recebendo 2 args, linhas e colunas
# ":" indica todos, no caso todas as linhas e "1:4" indicca das colunas
# 1 a 3 (a 4 não entra pq é a classe, mas colocamos como limite)
# não foi pega a primeira coluna "0" pois refere-se a id e id não é interessante
# como atributo previsor, ou seja, não costuma influenciar no resultado, é apenas
# uma identificação, chave, mas não serve como previsor.
previsores = base.iloc[:,1:4].values

# qnd colocamos só um valor sem o intervalo, então esse valor não é o limite
# e então está incluso. neste caso a classe usará todas as linhas ":" e 
# a coluna 4
classe = base.iloc[:,4].values 

# estamos chamando um valor de entrada que vai corrigir os valores faltantes
from sklearn.preprocessing import Imputer

# selecionando Imputer() e pressionando ctrl + i vemos a ajuda que ensina a usar
# o método

# por padrão missing_values já é 'NaN' então não precisava colocar
# fiz só por demonstração dos argumentos completos
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])
# após estes passos os valores faltantes foram preenchidos pela média de forma 
# automática

# ESCALONAMENTO DE ATRIBUTOS (deixar todos na mesma escala)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # criando um objeto da classe "StandardScaler"
previsores = scaler.fit_transform(previsores)

































































