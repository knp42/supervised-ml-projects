import streamlit as st
import requests

st.title("🛒 Moteur de Recommandation E-Commerce")
st.write("Entrez l'identifiant d'un client pour générer ses recommandations personnalisées.")

customer_id = st.number_input("ID Client (ex: 17850)", min_value=1, value=17850)
top_n = st.slider("Nombre de recommandations", min_value=1, max_value=10, value=5)

if st.button("Générer les recommandations"):
    response = requests.get(f"http://127.0.0.1:8000/recommend/{customer_id}?top_n={top_n}")
    
    if response.status_code == 200:
        data = response.json()
        st.success(f"Recommandations pour le client {customer_id} :")
        st.table(data['recommendations'])
    else:
        st.error("Client introuvable ou erreur de serveur.")