class OddsClassifier(): 
    def __init__(self):
        import numpy as np
        self.np = np
    
    def fit(self, previsores_treinamento, classe_treinamento):
        pass
    
    def predict(self, previsores_teste):
        
        previsoes = []
        for i in range(len(previsores_teste)):
            self.previsores_teste = previsores_teste[i][-3:]   
            indice = self.np.argmin(self.previsores_teste)
            
            if indice == 0:
                previsoes.append('H')
            elif indice == 1:
                previsoes.append('D')
            else:
                previsoes.append('A')
                
        return previsoes
        