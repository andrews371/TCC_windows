import pandas as pd
from sklearn.feature_selection import SelectFromModel

# métricas
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

base = pd.read_csv('dados_meio_tempo_com_odds.csv').drop(['Unnamed: 0'], axis=1).sample(frac=1).reset_index(drop=True)

previsores = base.iloc[:, 0:35].values
classe = base.iloc[:, 35].values
feat_labels = ['time_casa','time_visitante','golscasaht','golsvisitanteht','totalgolsht','possebolacasaht','possebolavisitanteht','chutescasaht','chutesvisitanteht','chutesnogolcasaht','chutesnogolvisitanteht','chutesforagolcasaht','chutesforagolvisitanteht','chutesbloqueadoscasaht','chutesbloqueadosvisitanteht','cornercasaht','cornervisitanteht','chutesdentroareacasaht','chutesdentroareavisitanteht','chutesforaareacasaht','chutesforaareavisitanteht','defesasgoleirocasaht','defesasgoleirovisitanteht','passescasaht','passesvisitanteht','passescertoscasaht','passescertosvisitanteht','duelosganhoscasaht','duelosganhosvisitanteht','disputasaereasvencidascasaht','disputasaereasvencidasvisitanteht','vencedorht','PSH','PSD','PSA']

classe_df = pd.DataFrame(classe)
previsores_df = pd.DataFrame(previsores)

#from sklearn.preprocessing import LabelEncoder
#labelencoder_X = LabelEncoder()
#previsores[:, 0] = labelencoder_X.fit_transform(previsores[:, 0])
#previsores[:, 1] = labelencoder_X.fit_transform(previsores[:, 1])
#previsores[:, 31] = labelencoder_X.fit_transform(previsores[:, 31])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0, 1, 31])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores)
previsores_onehotencoder_df = pd.DataFrame(previsores)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#previsores1 = scaler.fit_transform(previsores)
#previsores_standardscaler_df = pd.DataFrame(previsores)

'''previsores1 = pd.DataFrame(previsores1)
previsores1.drop(columns=[32, 33, 34], inplace=True)
previsores = pd.DataFrame(previsores)
previsores.drop(previsores.iloc[:,0:32],axis=1,inplace=True)
previsores = pd.concat([previsores1, previsores], axis=1, join='inner')
previsores = previsores.to_numpy()
previsores_df = pd.DataFrame(previsores)'''
pass

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
from odds import OddsClassifier

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
        classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        #classificador = OddsClassifier() 
        #classificador = MLPClassifier(verbose = True, max_iter = 1000,
                              #tol = 0.000010, solver='adam',
                              #hidden_layer_sizes=(100), activation = 'relu',
                              #batch_size=200, learning_rate_init=0.001)
        
        # treinamento
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])    
        # teste
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = metrics.accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)
   
resultados30 = np.asarray(resultados30)    

# visializar as previsoes
previsoes_df = pd.DataFrame(previsoes)

for i in range(resultados30.size):
    print(resultados30[i])
    
print(resultados30.mean())






