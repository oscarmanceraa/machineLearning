import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

class modeel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = None
        self.dataset_info = None
        self.feature_info = None
        
    def cargaData(self):
        """Carga y prepara los datos"""
        # Cargar dataset
        df = pd.read_csv('datos.csv')
        
        # Información del dataset
        self.dataset_info = {
            'num_records': len(df),
            'features': list(df.columns),
            'target_distribution': dict(df['TenYearCHD'].value_counts())
        }
        
        # Imputar valores faltantes - Actualizado para incluir todas las columnas
        self.imputer = SimpleImputer(strategy='mean')
        cols_to_impute = ['education', 'BPMeds', 'prevalentStroke', 
                         'cigsPerDay', 'totChol', 'BMI', 'heartRate', 'glucose']
        df[cols_to_impute] = self.imputer.fit_transform(df[cols_to_impute])
        
        # División de datos en train y test
        X = df.drop('TenYearCHD', axis=1)
        y = df['TenYearCHD']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        
        # Estandarización
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def train(self):
        """Entrena el modelo"""
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate(self):
        """Evalúa el modelo y genera métricas"""
        y_pred = self.model.predict(self.X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        # Generar matriz de confusión
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), 
                    annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.savefig('static/confusion_matrix.png')
        plt.close()
    
    def predict(self, input_data):
        """Realiza una predicción"""
        input_df = pd.DataFrame([input_data], columns=[
            'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 
            'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
            'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
        ])
        
        # Imputar valores faltantes
        cols_to_impute = ['education', 'BPMeds', 'prevalentStroke', 
                         'cigsPerDay', 'totChol', 'BMI', 'heartRate', 'glucose']
        input_df[cols_to_impute] = self.imputer.transform(input_df[cols_to_impute])
        
        scaled = self.scaler.transform(input_df)
        
        prediction = self.model.predict(scaled)[0]
        probability = self.model.predict_proba(scaled)[0][1]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'risk_level': 'Alto' if prediction == 1 else 'Bajo',
            'message': 'Riesgo alto de enfermedad cardiaca en 10 años' if prediction == 1 
                      else 'Riesgo bajo de enfermedad cardiaca en 10 años'
        }
    
    def inicio(self):
        """Carga datos, entrena y evalúa el modelo"""
        self.cargaData() 
        self.train()
        self.evaluate()