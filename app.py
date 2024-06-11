import streamlit as st
import requests
import pandas as pd

st.title("Machine Learning API Interaction")

st.header("Entraîner le modèle")
uploaded_file = st.file_uploader("Choisissez un fichier CSV pour l'entraînement", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
    if st.button("Entraîner le modèle"):
        response = requests.post("http://localhost:8000/training", json={"data": data.iloc[:, :-1].values.tolist(), "target": data.iloc[:, -1].values.tolist()})
        if response.status_code == 200:
            st.write(response.json())
        else:
            st.error(f"Erreur lors de l'entraînement du modèle: {response.text}")

st.header("Faire une prédiction")
input_data = st.text_area("Entrez les données pour la prédiction (format JSON)", "[[val1, val2, ...], [val1, val2, ...]]")
if st.button("Prédire"):
    response = requests.post("http://localhost:8000/predict", json={"data": eval(input_data)})
    if response.status_code == 200:
        st.write(response.json())
    else:
        st.error(f"Erreur lors de la prédiction: {response.text}")
