import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Recommandation E-Commerce", layout="centered")

st.title("🛒 Moteur de Recommandation E-Commerce")
st.write("Entrez l'identifiant d'un client pour générer ses recommandations personnalisées.")

# 1. Chargement du modèle depuis le fichier pkl hébergé sur GitHub
@st.cache_resource
def load_data():
    with open('recommender_model.pkl', 'rb') as f:
        return pickle.load(f)

data = load_data()
model = data['model']
user_to_index = data['user_to_index']
index_to_item = data['index_to_item']
user_items_matrix = data['user_items_matrix']
df_products = data['df_clean']

# 2. Formulaire utilisateur
customer_id = st.number_input("ID Client (ex: 17850)", min_value=1, value=17850)
top_n = st.slider("Nombre de recommandations", min_value=1, max_value=10, value=5)

if st.button("Générer les recommandations"):
    if customer_id not in user_to_index:
        st.error("Client inconnu (Cold Start).")
    else:
        user_idx = user_to_index[customer_id]
        
        # Génération des recommandations en direct
        ids, scores = model.recommend(
            userid=user_idx, 
            user_items=user_items_matrix[user_idx], 
            N=top_n
        )
        
        recs = []
        for item_idx, score in zip(ids, scores):
            code = index_to_item[item_idx]
            desc_matches = df_products[df_products['StockCode'] == code]['Description']
            desc = desc_matches.iloc[0] if not desc_matches.empty else "N/A"
            recs.append({"stock_code": str(code), "description": desc, "score": round(float(score), 4)})
            
        st.success(f"Recommandations pour le client {customer_id} :")
        st.table(pd.DataFrame(recs))