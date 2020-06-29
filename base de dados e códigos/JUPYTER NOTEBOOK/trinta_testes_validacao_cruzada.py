class TrintaTestes():

    def __init__(self, classificador, previsores, classe):
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        from sklearn.metrics import accuracy_score, confusion_matrix

        self.acuracia_final = []
        self.matriz_final = []
        self.StratifiedKFold = StratifiedKFold
        
        for i in range(30):
            kfold = self.StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
            acuracia_parcial = []
            matriz_parcial = []

            for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(classe.shape[0], 1))):

                # treinamento
                classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])    
                
                # teste
                previsoes = classificador.predict(previsores[indice_teste])
                
                # verificação de acurácia
                acuracia_parcial.append(accuracy_score(classe[indice_teste], previsoes))            
            
                # matriz de confusão
                matriz_parcial.append(confusion_matrix(classe[indice_teste], previsoes))
                

            # verificação de acurácia
            acuracia_parcial = np.asarray(acuracia_parcial)
            self.acuracia_final.append(acuracia_parcial.mean())
            
            # matriz de confusão
            self.matriz_final.append(np.mean(matriz_parcial, axis = 0))
        
        self.acuracia_final = np.asarray(self.acuracia_final)
        self.matriz_final = np.mean(self.matriz_final, axis = 0)

    def acuracia(self):        
        return self.acuracia_final

    def matrizConfusao(self):
        return self.matriz_final
            