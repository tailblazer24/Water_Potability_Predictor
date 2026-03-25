import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title("Water Potability Predictor")
st.write("Enter the water parameters below to check if the water is safe to use.")

# Load the trained model and scaler
with open('water_quality_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

feature_order = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 
                 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Create input widgets
inputs = {}
col1, col2 = st.columns(2)
with col1:
    inputs['ph'] = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
    inputs['Hardness'] = st.number_input("Hardness (mg/L)", min_value=0.0, value=200.0)
    inputs['Solids'] = st.number_input("Total Dissolved Solids (ppm)", min_value=0.0, value=20000.0)
    inputs['Chloramines'] = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0)
    inputs['Sulfate'] = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0)
with col2:
    inputs['Conductivity'] = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=400.0)
    inputs['Organic_carbon'] = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=15.0)
    inputs['Trihalomethanes'] = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, value=70.0)
    inputs['Turbidity'] = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0)

# When the user clicks the button
if st.button("Predict Potability"):
    # Create a DataFrame with one row
    input_df = pd.DataFrame([inputs])[feature_order]  # ensure correct order
    # Scale the input
    scaled_input = scaler.transform(input_df)
    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]  # probability of class 1
    
    # Display result
    if prediction == 1:
        st.success(f" Water is potable (safe) with {probability:.2%} confidence.")
    else:
        st.error(f" Water is NOT potable (unsafe). Risk probability: {probability:.2%}")
    
    # Show feature importance (optional)
    st.subheader("Why this prediction?")
    # Load feature importance from the model
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_order, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feat_imp, ax=ax)
    ax.set_title("Global Feature Importance")
    st.pyplot(fig)