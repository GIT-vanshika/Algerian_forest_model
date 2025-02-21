from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('model/ridge.pkl','rb'))
standard_scaler = pickle.load(open('model/scaler.pkl','rb'))

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Debugging: Print received values
            print(f"Received Input: {Temperature}, {RH}, {Ws}, {Rain}, {FFMC}, {DMC}, {ISI}, {Classes}, {Region}")

            # Scale input
            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            
            # Make prediction
            result = ridge_model.predict(new_data_scaled)
            print(f"Prediction: {result[0]}")  # Debugging output
            
            return render_template('home.html', results=result[0])

        except Exception as e:
            print(f"Error: {str(e)}")  # Print error in Flask console
            return render_template('home.html', results="Error occurred!")

    return render_template('home.html', results=None)
  

  
if __name__ == '__main__':
  application.run(host="0.0.0.0")
  