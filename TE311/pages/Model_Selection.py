import streamlit as st
from services.random_forest_selection import random_forest_selection
from dataset import load_data
from services.neural_networks_selection import neural_networks_selection
from services.support_vector_machine_selection import support_vector_machine_selection

# Model Selection Section
data = load_data(file_path=r'D:\csv file\disaster_sea.csv')

st.subheader("Model Selection")
model_selection = st.selectbox("Select a machine learning model:", ["Choose Model", "Random Forest", "Neural Networks", "Support Vector Machine", "K-Nearest Neighbors", ])

if model_selection == "Random Forest":
    random_forest_selection(data)
    
elif model_selection == "Neural Networks":
    neural_networks_selection(data)

elif model_selection == "Support Vector Machine":
    support_vector_machine_selection(data)

elif model_selection == "K-Nearest Neighbors":
    st.write("K-Nearest Neighbors is under development.")

elif model_selection == "Choose Model":
    st.write("Please select a model.")