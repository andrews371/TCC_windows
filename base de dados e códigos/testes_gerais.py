# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:49:41 2020

@author: André
"""

import pandas as pd
import numpy as np

base_com_odds = pd.read_csv('base_com_odds.csv')

teste = base_com_odds.loc[:,('PSH','PSD','PSA')]
teste = np.asarray(teste)
teste
teste.flatten()
a = np.argmin(teste.flatten())
print(a)
teste.flatten()[a]

teste = pd.DataFrame(teste)
teste.loc[:,'psh'] = 0
teste.drop(['psh'],inplace=True, axis=1)
teste.loc[:,0]

# adiconando/modificando nome de colunas em um dataframe 
teste.rename(columns={0:'psh', 1:'psd', 2:'psa'}, inplace=True)

teste2 = base_com_odds.loc[0:0,(['PSH','PSD','PSA'])]
teste2 = np.asarray(teste2)
teste2
np.argmax(teste2)
teste2.flatten()[2]

class Cachorro():
    def __init__(self, raca, altura, cor): # aqui podemos ter quantos atributos quisermos como parâmetro dessa função  
        self.raca = raca # esses são atributos do método
        self.altura = altura
        self.cor = cor
        
dog1 = Cachorro('pitbull',1.02,'preto')
dog1.raca
dog1.altura
dog1.cor

dog2 = Cachorro('pastor alemão',1.05,'amarelo')
dog2.raca
dog2.altura
dog2.cor

class Cachorro2():
    
    # aqui é qnd tiver algo que não mude, que seja igual pra todos os objetos
    # como raio por exemplo
    valor = ' OK'
    
    # init é o construtor que não é obrigado a classe ter
    # e como os outros métodos não precisa receber argumentos (mas sempre precisa
    # do self)
    def __init__(self, raca, cor):
        self.raca = raca + self.valor
        self.cor = cor
        
    def latir(self, altura): 
        self.altura = altura * 3
        print(f'Au Au minha raça é {self.raca} e minha altura é {self.altura}.')
        print(f'Au Au minha raça é {self.raca} e minha altura é {altura}.')
    
    # mesmo que o método não receba atributos, 1 nome como exemplo "self" (só um padrão) é necessário    
    def dormir(self): 
        print(f'Cachorro dormindo.')
        print(f'Raça do cachorro é {self.raca}')
        return self.raca
   

dog3 = Cachorro2('pitbull','preto')
dog3.raca
dog3.cor
dog3.latir(1.10)
dog3_dormir = dog3.dormir()
print(dog3.dormir())
print(dog3_dormir)

# ver o endereço da variável
a = 3
b = 3
print(hex(id(a)))
print(hex(id(b)))

test = []
test.append(0)
print(test[0])


# OUTRA FORMA DE BUSCAR CLASSIFICADORES MAIS IMPORTANTES
# O RETORNO SÃO OS ÍNDICES

# classificador é quem recebe o algoritmo de machine learning
'''importancias = classificador.feature_importances_
classificador = np.argsort(importancias)[::-1] # o ::-1 inverte a ordem colocando em ordem descendente
print(classificador)

# resgatando o nome das colunas pelos índices - modo 1
indices = base.iloc[:,[31,2]].columns
print(indices)

# resgatando o nome das colunas pelos índices - modo 2
indices = base.iloc[:,classificador].columns'''
pass


































