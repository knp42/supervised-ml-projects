from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Credit Card Fraud Detection API")

# Chargement des artefacts
with open('fraud_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']
features = data['features']

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Conversion en DataFrame
    input_data = pd.DataFrame([transaction.dict()])
    
    # Prétraitement de Time et Amount
    input_data['scaled_amount'] = scaler.transform(input_data[['Amount']].values)
    input_data['scaled_time'] = scaler.transform(input_data[['Time']].values)
    input_data = input_data.drop(['Time', 'Amount'], axis=1)
    
    # Re-ordonner les colonnes pour correspondre à l'entraînement
    input_data = input_data[features]
    
    # Prédictions
    proba = float(model.predict_proba(input_data)[0][1])
    is_fraud = bool(proba > 0.5)
    
    return {
        "fraud_prediction": is_fraud,
        "fraud_probability": round(proba, 4),
        "status": "ALERT" if is_fraud else "APPROVED"
    }