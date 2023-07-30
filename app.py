import numpy as np
from flask import Flask, request, render_template
import pickle
app=Flask(__name__,template_folder='templates')
model=pickle.load(open('C:/Users/M9bin/OneDrive/Documents/brainstroke/models/model.pkl','rb'))

@app.route('/')
def home():
    return render_template('brain.html')
@app.route('/predict',methods=['POST'])
def predict():
    features = request.form.values()
    processed_features = []

    for i in features:
        try:
            # Try converting to float (integers will also be converted to float)
            processed_features.append(float(i))
        except ValueError:
            # If it's not a valid float, keep it as a string (for categorical features)
            processed_features.append(i)

    features = [np.array(processed_features)]
    prediction = model.predict(features)
    output = round(prediction[0][0], 3)  # Assuming the model output is a 2D array

    return render_template('brain.html', prediction_text='The prediction value is {}'.format(output))
