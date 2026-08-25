import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Dashboard Ventes", layout="wide")

st.title("📊 Tableau de Bord des Ventes")
st.markdown("Analysez vos performances commerciales en un coup d'œil.")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data(file=None):
    if file is not None:
        return pd.read_csv(file)
    else:
        # Données de démonstration si aucun fichier n'est téléversé
        np.random.seed(42)
        dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
        villes = ["Douala", "Yaoundé", "Bafoussam", "Garoua"]
        produits = ["Produit A", "Produit B", "Produit C"]
        
        data = {
            "Date": np.random.choice(dates, 200),
            "Ville": np.random.choice(villes, 200),
            "Produit": np.random.choice(produits, 200),
            "Ventes_FCFA": np.random.randint(5000, 50000, 200),
            "Quantite": np.random.randint(1, 10, 200)
        }
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

# Barre latérale pour import de fichier et filtres
st.sidebar.header("⚙️ Options & Filtres")
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])
df = load_data(uploaded_file)

# Filtres dynamiques
villes_selectionnees = st.sidebar.multiselect(
    "Filtrer par Ville :",
    options=df["Ville"].unique(),
    default=df["Ville"].unique()
)

df_filtered = df[df["Ville"].isin(villes_selectionnees)]

# --- INDICATEURS CLÉS (KPIs) ---
total_ventes = df_filtered["Ventes_FCFA"].sum()
total_articles = df_filtered["Quantite"].sum()
panier_moyen = total_ventes / len(df_filtered) if len(df_filtered) > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Chiffre d'Affaires Total", f"{total_ventes:,.0f} FCFA")
col2.metric("Articles Vendus", f"{total_articles:,}")
col3.metric("Panier Moyen", f"{panier_moyen:,.0f} FCFA")

st.markdown("---")

# --- GRAPHIQUES INTERACTIFS ---
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📈 Évolution des Ventes")
    ventes_temps = df_filtered.groupby("Date")["Ventes_FCFA"].sum().reset_index()
    fig_line = px.line(ventes_temps, x="Date", y="Ventes_FCFA", title="Ventes dans le temps")
    st.plotly_chart(fig_line, use_container_width=True)

with right_col:
    st.subheader("🏙️ Ventes par Ville")
    ventes_ville = df_filtered.groupby("Ville")["Ventes_FCFA"].sum().reset_index()
    fig_bar = px.bar(ventes_ville, x="Ville", y="Ventes_FCFA", color="Ville", title="Chiffre d'affaires par ville")
    st.plotly_chart(fig_bar, use_container_width=True)

# Tableau de données brutes
with st.expander("🔍 Voir les données brutes"):
    st.dataframe(df_filtered)