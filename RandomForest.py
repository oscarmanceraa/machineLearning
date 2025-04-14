import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class ModeloRandomForest:
    def __init__(self):
        self.modelo = None
        self.metricas = {}
        self.caracteristicas = ['radius_mean', 'texture_mean', 'symmetry_mean']
        self.objetivo = 'diagnosis'
        self.ruta_modelo = 'static/models/random_forest_model.pkl'
        
    def entrenar(self):
        # Cargar datos
        datos = pd.read_csv('cancerData.csv')
        
        datos[self.objetivo] = datos[self.objetivo].map({'M': 1, 'B': 0})
        
        X = datos[self.caracteristicas]
        y = datos[self.objetivo]
        
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        self.modelo.fit(X_entrenamiento, y_entrenamiento)
        
        y_pred = self.modelo.predict(X_prueba)
        
        self.metricas['precision'] = accuracy_score(y_prueba, y_pred)
        self.metricas['informe_clasificacion'] = classification_report(y_prueba, y_pred)
        self.metricas['matriz_confusion'] = confusion_matrix(y_prueba, y_pred)
        
        os.makedirs(os.path.dirname(self.ruta_modelo), exist_ok=True)
        joblib.dump(self.modelo, self.ruta_modelo)
        
        return self.metricas
    
    def predecir(self, datos):
        if self.modelo is None:
            if os.path.exists(self.ruta_modelo):
                self.modelo = joblib.load(self.ruta_modelo)
            else:
                self.entrenar()
        
        if isinstance(datos, list):
            if len(datos) == 3: 
                datos = pd.DataFrame([datos], columns=self.caracteristicas)
            else:  
                datos = pd.DataFrame(datos, columns=self.caracteristicas)
        
        predicciones = self.modelo.predict(datos)
        probabilidades = self.modelo.predict_proba(datos)
        
        resultados = []
        for i, pred in enumerate(predicciones):
            resultado = {
                'prediction': int(pred),
                'prediction_label': 'Maligno' if pred == 1 else 'Benigno',
                'probability': probabilidades[i][pred] * 100  # Probabilidad como porcentaje
            }
            resultados.append(resultado)
            
        return resultados

modelo_rf = ModeloRandomForest()

def inicializar_modelo():
    if not os.path.exists(modelo_rf.ruta_modelo):
        modelo_rf.entrenar()
    else:
        modelo_rf.modelo = joblib.load(modelo_rf.ruta_modelo)
    return modelo_rf
