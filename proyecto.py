import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y encoders
dt = joblib.load("modelo_arbol.pkl")
le_places = joblib.load("encoder_places.pkl")
le_type = joblib.load("encoder_type.pkl")

st.title("🏠 Predictor de precio de inmuebles - CDMX")
st.markdown("Ingresa las características del inmueble para estimar su valor de mercado.")

st.divider()

# Inputs del usuario
col1, col2 = st.columns(2)

with col1:
    alcaldia = st.selectbox("Alcaldía", options=sorted(le_places.classes_))
    tipo = st.selectbox("Tipo de inmueble", options=sorted(le_type.classes_))

with col2:
    metros = st.number_input("Superficie cubierta (m²)", min_value=10, max_value=1000, value=80, step=5)
    precio_m2 = st.number_input("Precio por m² (MXN)", min_value=1000, max_value=100000, value=20000, step=500)

st.divider()

# Predicción
if st.button("Estimar precio", use_container_width=True):
    alcaldia_enc = le_places.transform([alcaldia])[0]
    tipo_enc = le_type.transform([tipo])[0]

    X_pred = pd.DataFrame([[metros, precio_m2, alcaldia_enc, tipo_enc]],
                          columns=["surface_covered_in_m2", "price_per_m2", "places_enc", "type_enc"])

    precio_estimado = dt.predict(X_pred)[0]

    st.success(f"💰 Precio estimado: **${precio_estimado:,.0f} MXN**")

    # Contexto adicional
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Precio estimado (MXN)", f"${precio_estimado:,.0f}")
    with col4:
        st.metric("Precio estimado (USD)", f"${precio_estimado / 17.5:,.0f}")
