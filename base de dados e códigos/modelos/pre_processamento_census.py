# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:11:26 2020

@author: André
"""

import pandas as pd
base = pd.read_csv('census.csv')

# nessa nova base não vamos tratar valores inconsistentes pq aqui não tem
# já foi tratado
# e não vamos tratar valores faltantes pq já foi tratado sendo colocado
# uma interrogação


# AQUI VAMOS TRANSFORMAR AS VARIÁVEIS CATEGÓRICAS EM NUMÉRICAS


# temos muito mais variáveis categóricas. muitos algoritmos precisam dessa transformação

# dividindo os atributos da base em previsores e classe
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # o ultimo é p as 
# variaveis dummy
labelencoders_previsores = LabelEncoder()

# só uma demonstração na coluna 1
labels = labelencoders_previsores.fit_transform(previsores[:,1])

# mas o que vale é esta linha em que vamos fazer a mesma coisa e colocar dentro 
# da própria variável (tabela) "previsores" mas não substituindo a tabela toda
# apenas a coluna 1 (workclass) já que o código é justamente para essa coluna 1

previsores[:,1] = labelencoders_previsores.fit_transform(previsores[:,1]) 

# agora fazemos o mesmo para todos os outros atributos categóricos
previsores[:,3] = labelencoders_previsores.fit_transform(previsores[:,3]) 
previsores[:,5] = labelencoders_previsores.fit_transform(previsores[:,5]) 
previsores[:,6] = labelencoders_previsores.fit_transform(previsores[:,6]) 
previsores[:,7] = labelencoders_previsores.fit_transform(previsores[:,7]) 
previsores[:,8] = labelencoders_previsores.fit_transform(previsores[:,8]) 
previsores[:,9] = labelencoders_previsores.fit_transform(previsores[:,9]) 
previsores[:,13] = labelencoders_previsores.fit_transform(previsores[:,13]) 

onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13]) 
previsores = onehotencoder.fit_transform(previsores).toarray()

# como a classe tbm é categórica tbm vamos transfromá-la em numérica
# sem afetar no cálculo dos algoritmos assim como fizemos antes
# aqui não será preciso usar o OneHotEncoder pq como se trata de classe
# é apenas uma classificação em 0 ou 1, ou 0,1,2... mas isso não será usado
# no cálculo dos algoritmos como calcular distâncias por exemplo, então isso
# não afetará o resultado da previsão

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)



# AGORA VAMOS FAZER O ESCALONAMENTO ASSIM COMO FIZEMOS NA BASE DE DADOS
# "CREDIT_DATA.CSV"


# relembrando que isso é fundamental para o funcionamento de alguns algoritmos
# e torna o processamento mais rápido mesmo se o algoritmo usado não precisar 
# dessa transformação

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

























































