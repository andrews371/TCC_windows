import pandas as pd

# métricas
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools

# plotagem
import matplotlib.pyplot as plt


base = pd.read_csv('dados_meio_tempo_com_odds.csv').drop(['Unnamed: 0'], axis=1).sample(frac=1).reset_index(drop=True)
#base_completa = pd.read_csv('dados_durante_jogo_pre_processados.csv')
#base = base_completa.copy()

'''base.drop(columns=['pais','campeonato','temporada','golscasaft','golsvisitanteft',
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
'''
pass

previsores = base.iloc[:, 0:35].values
classe = base.iloc[:, 35].values
target_names = ['H','D','A']

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

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#previsores = scaler.fit_transform(previsores)

from sklearn.ensemble import RandomForestClassifier
import numpy as np


# exemplo do uso de np.zeros e shape
# cria um vetor com 5 posições iniciando todas com zero
#teste1 = np.zeros(5)
# cria uma matriz com 5 posições e 1 coluna iniciando todas as posições com zero
#teste2 = np.zeros(shape=(5,1))

# agora vamos usar esse raciocínio para os previsores
#previsores.shape # aqui mostra a estrutura de previsores
#previsores.shape[0] # na posição 0 temos o número de linhas
# demonstração da utilização de np.zeros e shape
#b = np.zeros(shape=(previsores.shape[0], 1))

from sklearn.model_selection import StratifiedKFold
import view_functions

def plot_confusion_matrix(conf_matrix, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if classes:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

def calculo_matriz_confusao(cnf_matrix, classes_):
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    if classes_:
        plot_confusion_matrix(cnf_matrix, classes=classes_,
                      title='Confusion matrix, without normalization')
    else:
        plot_confusion_matrix(cnf_matrix,
                      title='Confusion matrix, without normalization')
        
    # Plot normalized confusion matrix
    plt.figure()
    if classes_:
        plot_confusion_matrix(cnf_matrix, classes=classes_, normalize=True,
                      title='Normalized confusion matrix')
    else:
        plot_confusion_matrix(cnf_matrix, normalize=True,
                      title='Normalized confusion matrix')

classes_=['Visitante', 'Empate', 'Mandante']
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
acuracia_lista = []
matrizes = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(classe.shape[0], 1))):
    # print('Índice treinamento: ', indice_treinamento, 'Índice teste: ', indice_teste)
    classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento]) 
    previsoes = classificador.predict(previsores[indice_teste])
    acuracia_parcial = accuracy_score(classe[indice_teste], previsoes)
    cm = confusion_matrix(classe[indice_teste], previsoes)
    matrizes.append(calculo_matriz_confusao(confusion_matrix(classe[indice_teste], previsoes),\
                    classes_))
    acuracia_lista.append(acuracia_parcial)
    
matriz_final = np.mean(matrizes, axis = 0) # média das 10 matrizes de confusão  
view_functions.plot_confusion_matrix(matriz_final, target_names)

acuracia_lista = np.asarray(acuracia_lista)
acuracia_media = acuracia_lista.mean() 
acuracia_std = acuracia_lista.std()

print(f'acurácia média: {acuracia_media}')
print(f'acurácia std: {acuracia_std}')
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

previsores_df = pd.DataFrame(previsores)
classe_df = pd.DataFrame(classe)