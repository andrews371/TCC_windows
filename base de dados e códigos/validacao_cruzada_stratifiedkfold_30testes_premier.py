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

#from sklearn.preprocessing import LabelEncoder
#labelencoder_X = LabelEncoder()
#previsores[:, 0] = labelencoder_X.fit_transform(previsores[:, 0])
#previsores[:, 1] = labelencoder_X.fit_transform(previsores[:, 1])
#previsores[:, 31] = labelencoder_X.fit_transform(previsores[:, 31])

#labelencoder_classe = LabelEncoder()
#classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0, 1, 31])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.ensemble import RandomForestClassifier
import numpy as np


# exemplo do uso de np.zeros e shape
# cria um vetor com 5 posições iniciando todas com zero
teste1 = np.zeros(5)
# cria uma matriz com 5 posições e 1 coluna iniciando todas as posições com zero
teste2 = np.zeros(shape=(5,1))

# agora vamos usar esse raciocínio para os previsores
previsores.shape # aqui mostra a estrutura de previsores
previsores.shape[0] # na posição 0 temos o número de linhas
# utilizando np.zeros e shape
b = np.zeros(shape=(previsores.shape[0], 1))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


resultados30 = []

for i in range(30):
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = i)
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores,
                                                        np.zeros(shape=(previsores.shape[0], 1))):
        # print('Índice treinamento: ', indice_treinamento, 'Índice teste: ', indice_teste)
        classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento]) 
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
    
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean() # acurácia da média das 10 matrizes de confusão  
    resultados30.append(media)

resultados30 = np.asarray(resultados30)
for i in range(resultados30.size):
    print(resultados30[i])

