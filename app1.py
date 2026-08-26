app = FastAPI(title="E-Commerce Recommender API")

# Chargement du modèle au démarrage
with open('recommender_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import numpy as np

user_to_index = data['user_to_index']
index_to_item = data['index_to_item']
user_items_matrix = data['user_items_matrix']
df_products = data['df_clean']

@app.get("/")
def home():
    return {"message": "API de Recommandation E-commerce opérationnelle"}

@app.get("/recommend/{customer_id}")
def get_recommendations(customer_id: int, top_n: int = 5):
    if customer_id not in user_to_index:
        raise HTTPException(status_code=404, detail="Client non trouvé (Cold Start)")
    
    user_idx = user_to_index[customer_id]
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
        
    return {"customer_id": customer_id, "recommendations": recs}