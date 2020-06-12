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
# demonstração da utilização de np.zeros e shape
b = np.zeros(shape=(previsores.shape[0], 1))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
resultados = []
matrizes = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(classe.shape[0], 1))):
    # print('Índice treinamento: ', indice_treinamento, 'Índice teste: ', indice_teste)
    classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento]) 
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
    resultados.append(precisao)

matriz_final = np.mean(matrizes, axis = 0)
resultados = np.asarray(resultados)
resultados.mean() # acurácia da média das 10 matrizes de confusão  
resultados.std()

print(f'resultados.mean(): {resultados.mean()}')
print(f'resultados.std(): {resultados.std()}')
print('\n')

# fórmulas à mão
# acurácia
acc = (matriz_final[0][0] +  matriz_final[1][1] +  matriz_final[2][2])\
      / (matriz_final[0][0] +  matriz_final[1][1] +  matriz_final[2][2]\
      + matriz_final[0][1] +  matriz_final[0][2]\
      + matriz_final[1][0] +  matriz_final[1][2]\
      + matriz_final[2][0] +  matriz_final[2][1])
print(f'acurácia: {acc}')

# precisão
prec = ((matriz_final[0][0]/(matriz_final[0][0] + matriz_final[1][0] + matriz_final[2][0]))\
       + (matriz_final[1][1]/(matriz_final[0][1] + matriz_final[1][1] + matriz_final[2][1]))\
       + (matriz_final[2][2]/(matriz_final[0][2] + matriz_final[1][2] + matriz_final[2][2])))\
       / 3
print(f'precisão: {prec}')

# recall
rec = ((matriz_final[0][0]/(matriz_final[0][0] + matriz_final[0][1] + matriz_final[0][2]))\
       + (matriz_final[1][1]/(matriz_final[1][0] + matriz_final[1][1] + matriz_final[1][2]))\
       + (matriz_final[2][2]/(matriz_final[2][0] + matriz_final[2][1] + matriz_final[2][2])))\
       / 3
print(f'recall: {rec}')

# f1-score
f1_score = (2 * prec * rec) / (prec + rec)
print(f'f1-score: {f1_score}')

previsoes_df = pd.DataFrame(previsoes)
classe_df = pd.DataFrame(classe)

