import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64  

# Dataset simulado: Temperatura  vs Consumo energético 
data = {
    "temperature": [37.45, 95.07, 73.19, 59.86, 15.60, 5.80, 86.61, 60.11, 70.80, 2.05, 96.99, 83.24, 21.23, 18.18, 30.42, 52.47, 43.19, 29.12, 61.18, 13.94],
    "consumption": [255.83, 179.09, 213.77, 199.60, 280.45, 295.09, 185.28, 216.20, 206.50, 297.98, 191.26, 189.52, 277.13, 278.60, 258.36, 232.90, 244.10, 254.82, 230.92, 284.13]
}

df = pd.DataFrame(data)  # Convertir a DataFrame
x = df[["temperature"]]  # Variable independiente
y = df[["consumption"]]  # Variable dependiente

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(x, y)

# Función para predecir consumo basado en la temperatura
def predict_consumption(temperature):
    temp_df = pd.DataFrame([[temperature]], columns=["temperature"])
    return model.predict(temp_df)[0][0]  

# Función para generar la gráfica de la regresión
def graph():
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color="blue", label="Datos reales")
    plt.plot(x, model.predict(x), color="red", label="Línea de regresión")
    plt.xlabel('Temperatura')
    plt.ylabel('Consumo')
    plt.title('Regresión Lineal: Temperatura vs Consumo')
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return image_base64  

# Prueba del modelo
if __name__ == "__main__":
    temperature = 50  
    predicted_consumption = predict_consumption(temperature)
    print(f"Predicción de consumo para {temperature}°C: {predicted_consumption:.2f}")
    graph()
