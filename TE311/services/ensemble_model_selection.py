import streamlit as st
from models.ensemble_model import train_ensemble_model
import pandas as pd
from calendar import monthrange

def disaster_prediction_selection(data):

    # Check if the dataset is empty
    if data.empty:
        st.error("The provided dataset is empty. Please upload a valid dataset.")
        return

    # User Inputs: Country
    country = st.selectbox("Select Country", data['Country'].unique())

    # User Inputs: Date Components
    st.write("### Enter Date")

    # Dropdown for Year
    year = st.selectbox("Select Year", list(range(1950, 2101)))

    # Dropdown for Month
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_name = st.selectbox("Select Month", month_names)

    # Convert month name to number
    month = month_names.index(month_name) + 1

    # Calculate the number of days in the selected month
    num_days = monthrange(year, month)[1]

    # Dropdown for Day
    day = st.selectbox("Select Day", list(range(1, num_days + 1)))

    # Button to predict disaster type
    if st.button("Predict Disaster Type"):
        try:
            # Prepare data for prediction
            prediction_input = {
                "Country": country,
                "Start Year": year,
                "Start Month": month,
                "Start Day": day
            }

            # Call the ensemble model
            results = train_ensemble_model(data, prediction_input)

            # Display Results
            st.write("## Prediction Results")
            predicted_disaster = results["predicted_disaster"]
            probability = results["prediction_probability"]

            # Display predictions
            st.success(f"**Predicted Disaster Type:** {predicted_disaster}")
            st.info(f"**Probability of Occurrence:** {probability:.2%}")

            # Additional Information
            st.write(
                f"Based on historical trends, the most likely disaster type in **{country}** "
                f"on **{year}-{month:02d}-{day:02d}** is **{predicted_disaster}**, "
                f"with a probability of **{probability:.2%}**."
            )

        except Exception as e:
            st.error(f"An error occurred while processing: {str(e)}")
