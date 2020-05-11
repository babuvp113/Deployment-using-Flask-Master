# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:55:57 2020

@author: ebabbpa
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__) #Initialising the flask
model = pickle.load(open('model.pkl','rb')) # Loading the ML model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    #Read the user input from the html GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html', prediction_text ='Employee Salary $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)