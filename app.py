import re
from datetime import datetime
from lineaRegLin import predict_consumption, graph
from flask import Flask, render_template, request
import lineaRegLin
from random import randint
import base64
import pandas as pd 
from RegresionLogistica import modeel
from RL import train_agent
#-------------------------------
app = Flask(__name__)

#---------Ruta main-----------------

@app.route('/')
def hello():
    return render_template("main.html")

#---------Ruta nombre-----------------

@app.route("/hello/<name>") 
def hello_there(name):
    now = datetime.now() 
    formatted_now = now.strftime("%A, %d %B, %Y at %X") 
    
    match_object = re.match("[a-zA-Z]+", name)
    
    if match_object :
        clean_name = match_object.group(0)
    else:
        clean_name = "friend"
        
    content = f"hello there {clean_name},  | the date is: {formatted_now}"
    return content

#---------Ruta Caso de exito-----------------

@app.route("/hello2")
def helloHTML():
        return render_template("casoExito.html")
    
#---------Ruta Regresion lineal-----------------

@app.route("/calc", methods=["GET", "POST"])
def calcular():
    result = None
    plot_image = None  

    if request.method == "POST":
        try:
            temperature = float(request.form['temperature'])  
            result = predict_consumption(temperature)  
            plot_image = graph()  

            if isinstance(plot_image, bytes):
                plot_image = base64.b64encode(plot_image).decode('utf-8')
                
        except KeyError:
            return "Error: No se proporcionó el campo 'temperature' en el formulario.", 400
        except ValueError:
            return "Error: El valor de temperatura no es válido.", 400

    print("plot_image:", plot_image[:50] if plot_image else "None")  

    return render_template("calcGrades.html", result=result, plot_image=plot_image)

#---------Ruta main-----------------

@app.route("/hello_form")
def hello_form():
    return render_template("hello_form.html")

#---------Ruta Mapa Regresion Logistica-----------------

@app.route("/mapa/")
def mapa():
    return render_template("mapa.html")
model = modeel()
model.inicio()  

#---------Ruta Regresion Logistica-----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('results.html', metrics=model.metrics)
        
    input_data = [
        float(request.form.get('male', 0)),
        float(request.form.get('age', 0)),
        float(request.form.get('education', 0)),
        float(request.form.get('currentSmoker', 0)),
        float(request.form.get('cigsPerDay', 0)),
        float(request.form.get('BPMeds', 0)),
        float(request.form.get('prevalentStroke', 0)),
        float(request.form.get('prevalentHyp', 0)),
        float(request.form.get('diabetes', 0)),
        float(request.form.get('totChol', 0)),
        float(request.form.get('sysBP', 0)),
        float(request.form.get('diaBP', 0)),
        float(request.form.get('BMI', 0)),
        float(request.form.get('heartRate', 0)),
        float(request.form.get('glucose', 0))
    ]
    
    result = model.predict(input_data)
    return render_template('results.html', result=result, metrics=model.metrics)


#---------Ruta Reforce learning-----------------
@app.route("/RL")
def  RLResult():
    acurracy, q_table = train_agent()
    return render_template("RL.html", acurracy=acurracy)
