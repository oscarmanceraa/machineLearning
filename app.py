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
from flask_mysqldb import MySQL
#-------------------------------
app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  
app.config['MYSQL_PASSWORD'] = ''  # cont db
app.config['MYSQL_DB'] = 'flaskbd'  # nombre db

# inicia mysql
mysql = MySQL(app)

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

#---------Ruta para mostrar datos de modelos de ML de MySQL-----------------
@app.route("/modelosML")
def modelosML():
    cur = mysql.connection.cursor()
    
    # Obtener el parámetro tipo de la URL
    tipo = request.args.get('tipo')
    
    # Si se especifica un tipo, filtrar por ese tipo
    if tipo:
        cur.execute("SELECT * FROM modelosml WHERE idTipoM = %s", [tipo])
    else:
        cur.execute("SELECT * FROM modelosml")
        
    results = cur.fetchall()
    
    # Obtener recursos gráficos para cada modelo
    recursos = {}
    for row in results:
        modelo_id = row[0]  # idModeloML
        cur.execute("SELECT idRecursoGrafico, RecursoGrafico, Recurso FROM recursosgraficos WHERE idModeloML = %s", [modelo_id])
        recursos_modelo = cur.fetchall()
        
        # Convertir las imágenes BLOB a base64 para mostrarlas en HTML
        recursos_procesados = []
        for recurso in recursos_modelo:
            if recurso[2]:  # Si hay datos en el campo Recurso (BLOB)
                # Convertir BLOB a base64
                imagen_base64 = base64.b64encode(recurso[2]).decode('utf-8')
                recursos_procesados.append({
                    'id': recurso[0],
                    'nombre': recurso[1],
                    'imagen': imagen_base64
                })
        
        recursos[modelo_id] = recursos_procesados
    
    cur.close()
    
    # Pasar el tipo seleccionado y los recursos a la plantilla
    return render_template("modelosML.html", results=results, tipo=tipo, recursos=recursos)
