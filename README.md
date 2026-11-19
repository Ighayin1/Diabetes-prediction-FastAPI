# Diabetes Prediction API

A machine learning API that predicts whether a person is likely to be diabetic, based on eight health measurements. The model is a linear Support Vector Machine trained on the Pima Indians Diabetes dataset and served with FastAPI. It is deployed on Heroku.

**Live API docs:** https://heroku-ml-api-3f418c83389f.herokuapp.com/docs

> **Disclaimer:** This is a learning project. Its output is a model prediction, not a medical diagnosis. Do not use it to make health decisions, and do not send real patient data to it.

## How it works

1. A client sends the eight input values as JSON to `POST /predict`.
2. The API scales the values with the same `StandardScaler` that was fitted during training.
3. The trained SVM classifies the input and the API returns the result.

## Model

| Detail | Value |
|---|---|
| Algorithm | Support Vector Classifier (`SVC`, linear kernel, balanced class weights) |
| Dataset | Pima Indians Diabetes (768 rows, 8 features) |
| Train/test split | 80% / 20%, stratified (614 train, 154 test) |
| Preprocessing | `StandardScaler` |
| Training accuracy | about 77.9% |
| Test accuracy | about 72.1% |

The model is a simple baseline. Accuracy is moderate, so treat its predictions as illustrative only.

## Input fields

| Field | Type | Description |
|---|---|---|
| `Pregnancies` | integer | Number of pregnancies |
| `Glucose` | integer | Plasma glucose concentration |
| `BloodPressure` | integer | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | integer | Triceps skin fold thickness (mm) |
| `Insulin` | integer | 2-hour serum insulin (mu U/ml) |
| `BMI` | float | Body mass index |
| `DiabetesPedigreeFunction` | float | Diabetes pedigree function (family history score) |
| `Age` | integer | Age in years |

## Example request

```bash
curl -X POST "https://heroku-ml-api-3f418c83389f.herokuapp.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 62,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.127,
    "Age": 47
  }'
```

Example response:

```json
{
  "prediction": "The person is diabetic"
}
```

You can also try the API in your browser using the interactive Swagger UI at `/docs`.

Note: the app runs on a Heroku Eco dyno, which sleeps when idle, so the first request after a quiet period may take a few seconds.

## Run locally

```bash
git clone https://github.com/Ighayin1/Diabetes-prediction-FastAPI.git
cd Diabetes-prediction-FastAPI
python -m venv venv
venv\Scripts\activate        # on macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs. To test the running server from Python, you can also run `api_implementation.py`.

## Project structure

```
.
├── main.py                 # FastAPI app with the /predict endpoint (deployed)
├── ml_api.py               # Earlier copy of the API used during development
├── api_implementation.py   # Example client that sends a request to the local API
├── diabetes_model.pkl      # Trained SVM model
├── scaler.pkl              # Fitted scaler used on the inputs
├── Diabetes prediction_Support vector machine.ipynb   # Model training notebook
├── diabetes.csv            # Training dataset
├── requirements.txt        # Python dependencies
├── Procfile                # Heroku process definition
└── .python-version         # Python version for Heroku
```

## Deployment

The app is deployed on Heroku and builds from this repository.

- `Procfile` starts the server with `uvicorn main:app --host 0.0.0.0 --port $PORT`
- `.python-version` selects the Python version (3.12)
- `requirements.txt` pins `scikit-learn`, `numpy`, `pandas` and `joblib` to the versions used in training, so the `.pkl` files load correctly

## Tech stack

Python, FastAPI, Uvicorn, scikit-learn, pandas, NumPy, joblib, Heroku



## Author

Osayi ([@Ighayin1](https://github.com/Ighayin1))
