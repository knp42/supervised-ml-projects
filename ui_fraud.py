import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Système de Détection de Fraude par Carte Bancaire")
st.write("Évaluation du risque de transaction en temps réel avec explicabilité.")

@st.cache_resource
def load_assets():
    with open('fraud_model.pkl', 'rb') as f:
        return pickle.load(f)

assets = load_assets()
model = assets['model']
scaler = assets['scaler']
features = assets['features']

st.subheader("Simulateur de Transaction")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Montant de la transaction ($)", min_value=0.0, value=149.99)
with col2:
    time_val = st.number_input("Temps (secondes écoulées)", min_value=0.0, value=406.0)

st.write("Seuil de détection configuré : **0.50**")

if st.button("Analyser la transaction"):
    # Construction d'un vecteur d'entrée factice (exemple de test)
    sample_vector = np.zeros((1, len(features)))
    df_sample = pd.DataFrame(sample_vector, columns=features)
    
    # Injection du montant et temps transformés
    df_sample['scaled_amount'] = scaler.transform([[amount]])
    df_sample['scaled_time'] = scaler.transform([[time_val]])
    
    proba = model.predict_proba(df_sample)[0][1]
    
    if proba >= 0.5:
        st.error(f"🚨 ALERTE FRAUDE DÉTECTÉE | Probabilité : {proba:.2%}")
    else:
        st.success(f"✅ TRANSACTION APPROUVÉE | Probabilité de fraude : {proba:.2%}")