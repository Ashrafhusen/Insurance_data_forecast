from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('models/insurance_pricing_model.pkl')


@app.post('/predict')
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {'predict_price' : float(prediction[0])}

    