import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import re

app = Flask(__name__)
regmodel = pickle.load(open('knnModel.pkl','rb'))
scalar   = pickle.load(open('norm.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

# create predict API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json["data"]
    print(data)
    # data = list(data)

    read_df_sample = pd.read_csv(data)
    features = [i for i in read_df_sample if re.findall("\As\d", i)]
    features.append('humidity')
    batch = read_df_sample.loc[read_df_sample['trial_state'] == 'exposure']
    batch = batch[features]
    batch = scalar.transform(batch)

    output = regmodel.predict(batch)
    print(output)
    return jsonify(output)

if __name__=="__main__":
    app.run(debug=True)