import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
previsores[:, 0] = labelencoder_X.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_X.fit_transform(previsores[:, 1])
previsores[:, 31] = labelencoder_X.fit_transform(previsores[:, 31])

#labelencoder_classe = LabelEncoder()
#classe = labelencoder_classe.fit_transform(classe)

#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0, 1, 31])],remainder='passthrough')
#previsores = column_tranformer.fit_transform(previsores)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

prev_df = pd.DataFrame(previsores)


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)

resultados = cross_val_score(classificador, previsores, classe, cv = 10)
print(resultados.mean())
print(resultados.std())

# VERIFICANDO ATRIBUTOS MAIS IMPORTANTES
classificador.fit(previsores,classe)

# imprimindo valores de importância independente da ordem e sem dar nomes aos atributos
print(classificador.feature_importances_)

# imprimindo valores dando nome aos atributos e em ordem 
columns_ = base.iloc[:, 0:32].columns
caracateristicas_importantes = pd.DataFrame(classificador.feature_importances_,
                                            index = columns_,
                                            columns = ['importance']).sort_values('importance', ascending = False)
print(caracateristicas_importantes)

# imprimindo valores com nome dos atributos, em ordem e através de gráfico em barras
caracateristicas_importantes.plot(kind='bar')


# BOXPLOT DOS RESULTADOS 
algoritmos = [0.5362, 0.5483, 0.5542, 0.5574, 0.5868, 0.6276, 0.6403]
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
plt.boxplot(algoritmos, patch_artist = True)
plt.title('Boxplot Premier League')
ax.set_xticklabels(['Algoritmos de Machine Learning'])
plt.show()
