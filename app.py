import pickle
from flask import Flask,request,app,jsonfy,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('knnModel.pkl','rb'))

@app.rout('/')
def home():
    return render_template('home.html')

# create predict API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)

