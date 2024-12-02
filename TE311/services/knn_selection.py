import streamlit as st
from models.k_nearest_neighbors import train_knn_model
from dataset import load_data
import pandas as pd

def knn_selection():
    st.title("KNN Disaster Prediction")

    # Load the dataset
    data = load_data(file_path = r'TE311/disaster_sea.csv')

    if data.empty:
        st.error("The dataset is empty. Please upload a valid dataset.")
        return

    # Select the disaster type
    st.write("### Select Disaster Type")
    disaster_type = st.selectbox("Disaster Type", data['Disaster Type'].unique())

    # Input K value
    k = st.slider("Number of Neighbors (K)", min_value=1, max_value=20, value=5)

    # Train and evaluate the KNN model
    if st.button("Train KNN Model"):
        try:
            results = train_knn_model(data, target='Disaster Type', k=k, disaster_type=disaster_type)

            st.write(f"## KNN Results for predicting '{disaster_type}'")
            st.success(f"**Model Accuracy:** {results['accuracy']:.2f}")

            # Display predictions
            st.write("### Predictions")
            predictions = pd.DataFrame({
                "Actual": results["y_test"],
                "Predicted": results["y_pred"]
            })
            st.dataframe(predictions)

            # Display classification report
            st.write("### Classification Report")
            st.text(results["classification_report"])
        except ValueError as e:
            st.error(f"Error: {e}")
