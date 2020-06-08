# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:02:19 2020

@author: André
"""

import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:,1:4])
previsores[:,1:4] = imputer.transform(previsores[:,1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


# DIVISÃO ENTRE BASE DE DADOS DE TREINAMENTO E BASE DE DADOS DE TESTE

from sklearn.model_selection import train_test_split                                                                
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# test_size=0.25 significa que 25% da base de dados será usada para teste e os outros 75%
# será usada para treinamento. geralmente deixamos mais dados no treinamento
# do que o teste


























