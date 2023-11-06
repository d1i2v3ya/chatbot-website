import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')


with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)


def convert_to_numeric(inputs):
    # Convert categorical variables to one-hot encoding
    gender = 1 if inputs['gender'] == 'Male' else 0
    ever_married = 1 if inputs['ever_married'] == 'Yes' else 0
    work_types = ['Private', 'Self-employed',
                  'Govt_job', 'children', 'Never_worked']
    work_type = work_types.index(inputs['work_type'])
    residence_types = ['Urban', 'Rural']
    residence_type = residence_types.index(inputs['Residence_type'])
    smoking_statuses = ['Unknown', 'formerly smoked', 'never smoked', 'smokes']
    smoking_status = smoking_statuses.index(inputs['smoking_status'])

    # Extract hypertension and heart_disease from inputs
    hypertension = inputs['hypertension']
    heart_disease = inputs['heart_disease']

    # Create a numeric input array
    numeric_inputs = np.array([
        gender,
        inputs['age'],
        hypertension,
        heart_disease,
        ever_married,
        work_type,
        residence_type,
        inputs['avg_glucose_level'],
        inputs['bmi'],
        smoking_status
    ]).reshape(1, -1)

    return numeric_inputs


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""

    if request.method == 'POST':
        # Retrieve form data with default values
        gender = request.form.get('gender', 'Male')
        age = int(request.form.get('age', 30))
        hypertension = int(request.form.get('hypertension', 0))
        heart_disease = int(request.form.get('heart_disease', 0))
        ever_married = request.form.get('ever_married', 'Yes')
        work_type = request.form.get('work_type', 'Private')
        residence_type = request.form.get('residence_type', 'Urban')
        avg_glucose_level = float(request.form.get('avg_glucose_level', 100.0))
        bmi = float(request.form.get('bmi', 25.0))
        smoking_status = request.form.get('smoking_status', 'Unknown')

        # Create a dictionary of inputs
        inputs = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        # Convert input values to numeric
        numeric_inputs = convert_to_numeric(inputs)

        # Make prediction using the KNN model
        prediction = knn_model.predict(numeric_inputs)

        # Set prediction result for display
        if prediction[0] == 1:
            prediction_text = "High chances of stroke"
        else:
            prediction_text = "Low chances of stroke"

    return render_template('home.html', prediction_text=prediction_text)



if __name__ == '__main__':
    app.run(debug=True)
