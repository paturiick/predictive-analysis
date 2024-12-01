import streamlit as st
import pandas as pd
from models.random_forest import train_random_forest_model

def random_forest_selection(data):
    # Select the disaster type within this function
    disaster_type = st.selectbox("Select Disaster Type to predict", data['Disaster Type'].unique())
    
    try:
        results = train_random_forest_model(data, disaster_type)
        
        if "error" in results:
            st.error(results["error"])
        else:
            # Display results
            st.write(f"### Results for predicting '{disaster_type}'")
            st.success(f"**Accuracy:** {results['accuracy']:.2f}")
            
            # Display classification report as a table
            st.write("### Classification Report")
            st.dataframe(results["classification_report"])

            # Display feature importances
            st.write("### Feature Importances")
            st.bar_chart(pd.Series(results["feature_importances"]))
    except Exception as e:
        st.error(f"An error occurred: {e}")
