# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:13:38 2020

@author: Andre
"""

import pandas as pd
import glob

# abrir vários arquivos do mesmo tipo e juntar vários dataframes
# em um só, formando um dataset
#file_list = [f for f in glob.glob("*.csv")]
#df_list = [pd.read_csv(f,encoding = 'unicode_escape') for f in file_list]
#final_dataset = pd.concat(df_list, sort=False).reset_index(drop=True)

#final_dataset.to_csv('dataset.csv')

final_dataset = pd.read_csv('dataset1.csv').drop(['Unnamed: 0'], axis=1).sample(frac=1).reset_index(drop=True)
total_jogos = len(final_dataset)

# calculando qual o percentual de vitória do time com qualquer grau de favoritismo

home_odd_menor = final_dataset.loc[(final_dataset['PSH'] < final_dataset['PSD'])\
                                   & (final_dataset['PSH'] < final_dataset['PSA'])]

draw_odd_menor = final_dataset.loc[(final_dataset['PSD'] < final_dataset['PSH'])\
                                   & (final_dataset['PSD'] < final_dataset['PSA'])]

away_odd_menor = final_dataset.loc[(final_dataset['PSA'] < final_dataset['PSH'])\
                                   & (final_dataset['PSA'] < final_dataset['PSD'])]


percentualHomeOddMenor = (len(home_odd_menor)/total_jogos) * 100
percentualDrawOddMenor = (len(draw_odd_menor)/total_jogos) * 100
percentualawayOddMenor = (len(away_odd_menor)/total_jogos) * 100


# calculando qual o percentual de acertos se sempre 
# previsse vitória do favorito - odd < 2

home_favorito = final_dataset.loc[final_dataset['PSH'] < 2]
draw_favorito = final_dataset.loc[final_dataset['PSD'] < 2]
away_favorito = final_dataset.loc[final_dataset['PSA'] < 2]

percetualHomeFavorito = (len(home_favorito)/total_jogos) * 100
percentualAwayFavorito = (len(away_favorito)/total_jogos) * 100

# contando qual o percentual de vitórias do time da casa,
# percentual de empate e de vitórias do time visitante

home_vitorias = final_dataset.loc[final_dataset['FTR'] == 'H']
draw_vitorias = final_dataset.loc[final_dataset['FTR'] == 'D']
away_vitorias = final_dataset.loc[final_dataset['FTR'] == 'A']

percentualHomeVitorias = (len(home_vitorias)/total_jogos) * 100
percentualDrawVitorias = (len(draw_vitorias)/total_jogos) * 100
percentualAwayVitorias = (len(away_vitorias)/total_jogos) * 100

# TESTES DE VALORES NULOS - NAN
# isnull = isna

#final_dataset.loc[:,'AwayTeam':'BbMxAHA'].isnull().sum()
#final_dataset.isnull().sum()
#teste_nan = final_dataset[final_dataset['Div'].isna()]
#teste_nan2 = final_dataset.loc[:,'Div'].isna().sum()
#print(teste_nan2)


