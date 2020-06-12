import pandas as pd

base_completa = pd.read_csv('dados_durante_jogo_pre_processados.csv')
base = base_completa.copy()

base.drop(columns=['pais','campeonato','temporada','golscasaft','golsvisitanteft',
                   'resultadoht','resultadoft','totalgolsft','possebolacasaft',
                   'possebolavisitanteft','chutescasaft','chutesvisitanteft',
                   'chutesnogolcasaft','chutesnogolvisitanteft','chutesforagolcasaft',
                   'chutesforagolvisitanteft','chutesbloqueadoscasaft','chutesbloqueadosvisitanteft',
                   'cornercasaft','cornervisitanteft','chutesdentroareacasaft','chutesdentroareavisitanteft',
                   'chutesforaareacasaft','chutesforaareavisitanteft','defesasgoleirocasaft',
                   'defesasgoleirovisitanteft','passescasaft','passesvisitanteft',
                   'passescertoscasaft','passescertosvisitanteft','duelosganhoscasaft',
                   'duelosganhosvisitanteft','disputasaereasvencidascasaft',
                   'disputasaereasvencidasvisitanteft'], axis=1, inplace=True)

previsores = base.iloc[:, 0:32].values
classe = base.iloc[:, 32].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0, 1, 31])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

resultados30 = []
for i in range(30):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(classe.shape[0], 1))):
        #classificador = GaussianNB()
        #classificador = DecisionTreeClassifier()
        #classificador = LogisticRegression()
        #classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0)
        #classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
        #classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        classificador = MLPClassifier(verbose = True, max_iter = 1000,
                              tol = 0.000010, solver='adam',
                              hidden_layer_sizes=(100), activation = 'relu',
                              batch_size=200, learning_rate_init=0.001)
        
        
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)
    
resultados30 = np.asarray(resultados30)    
resultados30.mean()
for i in range(resultados30.size):
    print(resultados30[i])

