# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:13:38 2020

@author: Andre
"""

import pandas as pd
import numpy as np
import glob

# abrir vários arquivos do mesmo tipo e juntar vários dataframes
# em um só, formando um dataset
#file_list = [f for f in glob.glob("*.csv")]
#df_list = [pd.read_csv(f,encoding = 'unicode_escape') for f in file_list]
#base_com_odds = pd.concat(df_list, sort=False).reset_index(drop=True)
#base_com_odds.to_csv('dataset.csv')

# SE A COLUNA FTR = H, ENTÃO CONTE A COLUNA DO 2o ARGUMENTO (FTR)
# A COLUNA DO 2o ARGUMENTO, NÃO PRECISA SER IGUAL A DO 1o, PODERIA SER QUALQUER UMA OUTRA)
# se não usar o 2o argumento, ele soma todas as colunas
# home_favorito = home_odds_menor.loc[home_odds_menor['FTR']=='H', 'FTR'].count()/total_jogos*100
# draw_favorito = draw_odds_menor.loc[home_odds_menor['FTR']=='D', 'FTR'].count()/total_jogos*100
# away_favorito = away_odds_menor.loc[home_odds_menor['FTR']=='A', 'FTR'].count()/total_jogos*100



base_com_odds = pd.read_csv('base_com_odds.csv').drop(['Unnamed: 0'], axis=1).sample(frac=1).reset_index(drop=True)
#base_sem_odds = pd.read_csv('base_sem_odds.csv').drop(['Unnamed: 0'], axis=1).sample(frac=1).reset_index(drop=True)


# calculando qual o percentual de vitória do time com qualquer grau de favoritismo
# calculando só para time jogando em casa, só para time jogando fora, só para empate

percentualHomeoddsMenor = (base_com_odds.loc[(base_com_odds['PSH'] < base_com_odds['PSD'])\
                                   & (base_com_odds['PSH'] < base_com_odds['PSA'])\
                                   & (base_com_odds['FTR'] == 'H'), 'FTR']\
                                   .count() / len(base_com_odds)) * 100

percentualDrawoddsMenor = (base_com_odds.loc[(base_com_odds['PSD'] < base_com_odds['PSH'])\
                                   & (base_com_odds['PSD'] < base_com_odds['PSA'])\
                                   & (base_com_odds['FTR'] == 'D'), 'FTR']\
                                   .count() / len(base_com_odds)) * 100
                                            
percentualAwayoddsMenor = (base_com_odds.loc[(base_com_odds['PSA'] < base_com_odds['PSH'])\
                                   & (base_com_odds['PSA'] < base_com_odds['PSD'])\
                                   & (base_com_odds['FTR'] == 'A'), 'FTR']\
                                   .count() / len(base_com_odds)) * 100
                                             
                                                                                          
# prevendo sempre vitória de quem tem a odds mais baixa, seja time jogando em casa 
# ou visitante ou prevendo empate 

percentualGeraloddsMenor = percentualHomeoddsMenor + percentualDrawoddsMenor\
                          + percentualAwayoddsMenor   


# simulando apostas em times com odds mais baixas

print('\n')
apostas_realizadas = 0
apostas_vencidas = 0
for key,row in base_com_odds.iterrows():
    if (row['PSH'] < row['PSA']) and (row['PSH'] < row['PSD']):
        apostas_realizadas += 1
        if row['FTR'] == 'H':
            apostas_vencidas += 1
    
    elif (row['PSA'] < row['PSH']) and (row['PSA'] < row['PSD']):
        apostas_realizadas += 1
        if row['FTR'] == 'A':
            apostas_vencidas += 1
            
percentual_acerto = (apostas_vencidas / apostas_realizadas) * 100
print(f'apostas realizadas {apostas_realizadas}')
print(f'apostas vencidas {apostas_vencidas}')
print(f'percentual de acerto {percentual_acerto}')
    

# contando qual o percentual de vitórias do time da casa,
# percentual de empate e de vitórias do time visitante

percentualHomeVitorias = (base_com_odds.loc[base_com_odds['FTR'] == 'H', 'FTR']\
                         .count() / len(base_com_odds)) * 100
                          
percentualDrawVitorias = (base_com_odds.loc[base_com_odds['FTR'] == 'D', 'FTR']\
                         .count() / len(base_com_odds)) * 100
                          
percentualAwayVitorias = (base_com_odds.loc[base_com_odds['FTR'] == 'A', 'FTR']\
                         .count() / len(base_com_odds)) * 100

# Impressões

'''for key1,row1 in base_sem_odds.iterrows():
    for key2,row2 in base_com_odds.iterrows():
        if (row1['time_casa'] == row2['HomeTeam'] and row1['time_visitante'] == row2['AwayTeam'])\
        and (row1['temporada'] == row2['temporada']):
            base_sem_odds.loc[key1, 'PSH'] = base_com_odds.loc[key2, 'PSH']
            base_sem_odds.loc[key1, 'PSD'] = base_com_odds.loc[key2, 'PSD']
            base_sem_odds.loc[key1, 'PSA'] = base_com_odds.loc[key2, 'PSA']

base_sem_odds.to_csv('dados_meio_tempo_com_odds.csv')                      
                          
# Adicionando nova coluna ao dataframe e fazendo adiçáo de valores

for key,row in base_com_odds.iterrows():
    if row['Date'] > '2015-07-31' and row['Date'] < '2016-06-01':
        base_com_odds.loc[key, 'temporada'] = '2015-2016'
    elif row['Date'] > '2016-07-31' and row['Date'] < '2017-06-01':
        base_com_odds.loc[key, 'temporada'] = '2016-2017'
    elif row['Date'] > '2017-07-31' and row['Date'] < '2018-06-01':
        base_com_odds.loc[key, 'temporada'] = '2017-2018'
    else:
        base_com_odds.loc[key, 'temporada'] = '2018-2019'

base_com_odds.to_csv('base_com_odds.csv')


# verificando todos os times existentes na base sem repetir  
# posso usar mais de uma coluna tbm se precisar
list(set(base_com_odds[['HomeTeam']].as_matrix().reshape((1,-1)).tolist()[0]))
list(set(base_sem_odds[['time_casa']].as_matrix().reshape((1,-1)).tolist()[0]))

for key,row in base_sem_odds.iterrows():
    if row['time_visitante'] == 'Swansea City':
        base_sem_odds.loc[key, 'time_visitante'] = 'Swansea'
        
    elif row['time_visitante'] == 'West Bromwich Albion':
        base_sem_odds.loc[key, 'time_visitante'] = 'West Brom'
        
    elif row['time_visitante'] == 'West Ham United':
        base_sem_odds.loc[key, 'time_visitante'] = 'West Ham'
        
    elif row['time_visitante'] == 'Newcastle United':
        base_sem_odds.loc[key, 'time_visitante'] = 'Newcastle'
        
    elif row['time_visitante'] == 'Manchester City':
        base_sem_odds.loc[key, 'time_visitante'] = 'Man City'
        
    elif row['time_visitante'] == 'Norwich City':
        base_sem_odds.loc[key, 'time_visitante'] = 'Norwich'
        
    elif row['time_visitante'] == 'Hull City':
        base_sem_odds.loc[key, 'time_visitante'] = 'Hull'
        
    elif row['time_visitante'] == 'Brighton & Hove Albion':
        base_sem_odds.loc[key, 'time_visitante'] = 'Brighton'
        
    elif row['time_visitante'] == 'Cardiff City':
        base_sem_odds.loc[key, 'time_visitante'] = 'Cardiff'
        
    elif row['time_visitante'] == 'Huddersfield Town':
        base_sem_odds.loc[key, 'time_visitante'] = 'Huddersfield'
        
    elif row['time_visitante'] == 'Manchester United':
        base_sem_odds.loc[key, 'time_visitante'] = 'Man United'
        
    elif row['time_visitante'] == 'Wolverhampton':
        base_sem_odds.loc[key, 'time_visitante'] = 'Wolves'
        
    elif row['time_visitante'] == 'Stoke City':
        base_sem_odds.loc[key, 'time_visitante'] = 'Stoke'
        
    elif row['time_visitante'] == 'Leicester City':
        base_sem_odds.loc[key, 'time_visitante'] = 'Leicester'
        
base_sem_odds.to_csv('base_sem_odds.csv')'''
pass


                          
# para deixar no ordem o índice de um dataframe criado a partir de outro
# usa-se ao final do comando de geração do novo dataframe:
# .reset_index(drop=True)

# TESTES DE VALORES NULOS - NAN
# isnull = isna

#base_com_odds.loc[:,'FTR'].isnull().sum()
#base_com_odds.isnull().sum()
#teste_nan = base_com_odds[base_com_odds['Div'].isna()]
#teste_nan2 = base_com_odds.loc[:,'Div'].isna().sum()
#print(teste_nan2)


