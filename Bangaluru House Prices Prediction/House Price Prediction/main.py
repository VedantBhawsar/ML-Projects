from flask import Flask, render_template, request
import pandas as pd
import os
import pickle
import numpy as np

app = Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    data=request.json
    location = data['location']
    bhk = data['bhk']
    bath = data['bath']
    sqft = data['total_sqft']    # print(data['location'])
    input = pd.DataFrame([[data['location'],data['bhk'], data['bath'], data['total_sqft'] ]], columns=['location', 'total_sqft','bhk', 'bath' ])
    prediction = pipe.predict(input)[0]
    return str(np.round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
