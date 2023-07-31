import streamlit as st
import pickle
import numpy as np

# Load the KNN model
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


def main():
    st.title("Stroke Prediction")

    # Input fields for user data
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox(
        "Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", value=100.0)
    bmi = st.number_input("BMI", value=25.0)
    smoking_status = st.selectbox(
        "Smoking Status", ["Unknown", "formerly smoked", "never smoked", "smokes"])

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

    # Create a submit button
    if st.button("Submit"):
        # Convert input values to numeric
        numeric_inputs = convert_to_numeric(inputs)

        # Make prediction using the KNN model
        prediction = knn_model.predict(numeric_inputs)

        # Show prediction result in a separate window
        st.write("Prediction Result:")
        if prediction[0] == 1:
            st.write("High chances of stroke")
        else:
            st.write("Low chances of stroke")


if __name__ == "__main__":
    main()
