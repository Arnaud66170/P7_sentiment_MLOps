import streamlit as st
import requests

st.set_page_config(page_title = "Analyse de sentiments", layout = "centered")

st.title("✈️ Air Paradis - Analyse de sentiment")
st.markdown("Saisis un tweet (en anglais) pour connaître son **sentiment prédictif**.")

# Zone de saisie
tweet = st.text_area("Entrer le tweet ici :", height = 150)

# Appel API
if st.button("Estimer le sentiment du tweet") and tweet.strip():
    try:
        response = requests.post(
            "http://13.38.46.11:8000/predict",
            json={"text": tweet}
        )
        if response.status_code == 200:
            result = response.json()
            sentiment = result["sentiment"]
            label = result["prediction_label"]

            if label == 1:
                st.success(f"👍 Sentiment **positif** détecté.")
            else:
                st.error(f"👎 Sentiment **négatif** détecté.")
        else:
            st.warning("L'API n'a pas répondu correctement.")

    except Exception as e:
        st.error(f"Erreur lors de l’appel à l’API : {e}")
