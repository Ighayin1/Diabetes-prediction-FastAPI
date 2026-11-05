import requests

url = 'http://127.0.0.1:8000/predict'

input_data_for_model = {
    'Pregnancies': 1,
    'Glucose': 14,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.3,
    'Age': 50
}

response = requests.post(url, json = input_data_for_model)
print(response.status_code)
print(response.text)
