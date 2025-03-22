import re
from datetime import datetime
from lineaRegLin import predict_consumption, graph
from flask import Flask, render_template, request
import lineaRegLin
from random import randint

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("main.html")

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

@app.route("/hello2")
def helloHTML():
        return render_template("index.html")

@app.route("/calc", methods=["GET", "POST"])
def calcular():
    result = None
    plot_image = None  

    if request.method == "POST":
        try:
            temperature = float(request.form['temperature'])  
            result = predict_consumption(temperature)  
            plot_image = graph()  # Generar la imagen de la gráfica

            if isinstance(plot_image, bytes):
                plot_image = base64.b64encode(plot_image).decode('utf-8')
                
        except KeyError:
            return "Error: No se proporcionó el campo 'temperature' en el formulario.", 400
        except ValueError:
            return "Error: El valor de temperatura no es válido.", 400

    print("plot_image:", plot_image[:50] if plot_image else "None")  

    return render_template("calcGrades.html", result=result, plot_image=plot_image)

if __name__ == "__main__":
    app.run(debug=True)
