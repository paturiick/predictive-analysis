import streamlit as st
from models.ensemble_model import train_ensemble_model
from streamlit_extras.dataframe_explorer import dataframe_explorer

def disaster_prediction_selection(data):
    st.title("Ensemble Model for Disaster Prediction")

    disaster_type = st.selectbox("Select Disaster Type to predict", data['Disaster Type'].unique())

    if data.empty:
        st.error("The provided dataset is empty. Please upload a valid dataset.")
        return

    try:
        results = train_ensemble_model(data, disaster_type)

        st.write(f"## Results for predicting '{disaster_type}'")
        st.success(f"**Ensemble Model Accuracy:** {results['accuracy']:.2f}")

        # Classification Report
        st.write("### Classification Report")
        st.dataframe(results["classification_report"])
        # Country-Specific Probabilities
        st.write("### Disaster Probability")
        country_probabilities = results["country_predictions"]
        filtered_df = dataframe_explorer(country_probabilities)
        st.dataframe(filtered_df, width=4000)

        # Explanation for Country-Specific Predictions
        st.info(
            f"The table above highlights the likelihood of '{disaster_type}' occurring in different areas of each countries. "
            f"Countries with higher probabilities are more vulnerable and may require immediate disaster management efforts."
        )
    except Exception as e:
        st.error(f"An error occurred while processing: {str(e)}")