import streamlit as st
import pandas as pd
from models.ensemble_model import train_ensemble_model

def disaster_prediction_selection(data):
    # Select the disaster type within this function
    disaster_type = st.selectbox("Select Disaster Type to predict", data['Disaster Type'].unique())

    # Ensure the dataset is not empty
    if data.empty:
        st.error("The provided dataset is empty. Please upload a valid dataset.")
        return

    try:
        # Call the ensemble model training and prediction function
        results = train_ensemble_model(data, disaster_type)

        # Display global model results
        st.write(f"## Results for predicting '{disaster_type}'")
        st.success(f"**Ensemble Model Accuracy:** {results['accuracy']:.2f}")

        # Classification Report
        st.write("### Classification Report")
        st.dataframe(results["classification_report"])

        # Feature Importances
        st.write("### Feature Importances (Random Forest)")
        feature_importances = pd.Series(results["feature_importances"]).sort_values(ascending=False)
        st.bar_chart(feature_importances)

        # Country-Specific Probabilities
        st.write("### Disaster Probability by Country")
        country_probabilities = results["country_predictions"]
        st.dataframe(country_probabilities, width=1000)
    except Exception as e:
        st.error(f"An error occurred while processing: {str(e)}")
