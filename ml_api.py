from fastapi import FastAPI
from pydantic import BaseModel #Basemodel sets format for which the data will be posted to API
import joblib
import pandas as pd

app = FastAPI()

#loading saved model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

class model_input(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post('/predict')
def predict(input_parameters: model_input):
    input_data = pd.DataFrame([[input_parameters.Pregnancies,
            input_parameters.Glucose,
            input_parameters.BloodPressure,
            input_parameters.SkinThickness,
            input_parameters.Insulin,
            input_parameters.BMI,
            input_parameters.DiabetesPedigreeFunction,
            input_parameters.Age]],
            columns=[
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ]
            )
    std_data = scaler.transform(input_data)

    prediction = model.predict(std_data)

    if prediction[0] == 1:
        return {'prediction': 'The person is diabetic'}
    else:
        return {'prediction': 'The person is non-diabetic'}