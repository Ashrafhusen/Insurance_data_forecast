import streamlit as st
import pandas as pd
import joblib


model = joblib.load('models/insurance_pricing_model.pkl')
st.title('Insurance Premium Predictor')

inputs = {f'V{i}' : st.number_input(f'Features V{i}' , 0 ) for i in range(1 , 86)}


if st.button('Prediction'):
    df = pd.DataFrame([inputs])
    predictions = model.predict(df)
    st.success(f"Predicted Insurance Price : ${predictions[0]:,.2f}")