import pandas as pd
from sklearn.feature_selection import SelectFromModel

base = pd.read_csv('dados_meio_tempo_com_odds.csv').drop(['Unnamed: 0'], axis=1).sample(frac=1).reset_index(drop=True)

previsores = base.iloc[:, 0:35].values
classe = base.iloc[:, 35].values
feat_labels = ['time_casa','time_visitante','golscasaht','golsvisitanteht','totalgolsht','possebolacasaht','possebolavisitanteht','chutescasaht','chutesvisitanteht','chutesnogolcasaht','chutesnogolvisitanteht','chutesforagolcasaht','chutesforagolvisitanteht','chutesbloqueadoscasaht','chutesbloqueadosvisitanteht','cornercasaht','cornervisitanteht','chutesdentroareacasaht','chutesdentroareavisitanteht','chutesforaareacasaht','chutesforaareavisitanteht','defesasgoleirocasaht','defesasgoleirovisitanteht','passescasaht','passesvisitanteht','passescertoscasaht','passescertosvisitanteht','duelosganhoscasaht','duelosganhosvisitanteht','disputasaereasvencidascasaht','disputasaereasvencidasvisitanteht','vencedorht','PSH','PSD','PSA']

classe_df = pd.DataFrame(classe)
previsores_df = pd.DataFrame(previsores)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0, 1, 31])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores)
previsores_onehotencoder_df = pd.DataFrame(previsores)

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#previsores = scaler.fit_transform(previsores)
#previsores_standardscaler_df = pd.DataFrame(previsores)

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
importancia_caracteristicas30 = []
for i in range(30):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
    resultados1 = []
    importancia_caracteristicas1 = []
    
    for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(classe.shape[0], 1))):
        #classificador = GaussianNB()
        #classificador = DecisionTreeClassifier()
        #classificador = LogisticRegression()
        #classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0)
        #classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
        #classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        classificador = OddsClassifier() 
        #classificador = MLPClassifier(verbose = True, max_iter = 1000,
                              #tol = 0.000010, solver='adam',
                              #hidden_layer_sizes=(100), activation = 'relu',
                              #batch_size=200, learning_rate_init=0.001)
        
        # treinamento
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        
        '''for feature in zip(feat_labels, classificador.feature_importances_):
            print(feature)            
        sfm = SelectFromModel(classificador, threshold=0.15)
        sfm.fit(previsores[indice_treinamento], classe[indice_treinamento])
        SelectFromModel(estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=40, n_jobs=-1, oob_score=False, random_state=0,
            verbose=0, warm_start=False),
        prefit=False, threshold=0.15)
        for feature_list_index in sfm.get_support(indices=True):
            print(feat_labels[feature_list_index])'''
        pass
    
        # teste
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)
    
    importancia_caracteristicas1 = np.asarray(importancia_caracteristicas1)
    importancia_caracteristicas30.append(importancia_caracteristicas1)
    
 # visializar os previsoes
previsoes_df = pd.DataFrame(previsoes)
   
resultados30 = np.asarray(resultados30)    
resultados30.mean()
for i in range(resultados30.size):
    print(resultados30[i])







