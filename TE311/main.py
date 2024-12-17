import streamlit as st
import os
import pandas as pd
from dataset import load_data
from services.random_forest_selection import disaster_prediction_selection
from services.neural_networks_selection import neural_networks_selection
from streamlit_option_menu import option_menu

# App Configuration
config_content = """
[theme]
primaryColor="#859f3d"
backgroundColor="#1a1a19"
secondaryBackgroundColor="#31511e"
textColor="#f6fcdf"
font="sans serif"
"""

st.set_page_config(layout="wide")
os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/config.toml", "w") as config_file:
    config_file.write(config_content)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dataset", "Model Selection"],
        icons=["file-earmark", "robot"],
        menu_icon="cast",
        default_index=0,
    )

# File Path
file_path = r'C:\Users\user\PycharmProjects\TE311/disaster_sea.csv'

if selected == "Dataset":
    st.title("Predictive Analysis for Natural Disaster Management")
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        st.success(f"Dataset loaded successfully from {file_path}")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.write(data.head())

    st.subheader("Dataset Statistics")
    st.write(data.describe())

    st.subheader("Filter Data")
    columns = data.columns.tolist()
    selected_columns = st.selectbox("Select columns to filter", columns)
    unique_values = data[selected_columns].unique()
    selected_value = st.selectbox("Select value to filter", unique_values)

    filtered_data = data[data[selected_columns] == selected_value]
    st.write(filtered_data)

elif selected == "Model Selection":
    try:
        data = load_data(file_path=file_path)
    except Exception as e:
        st.error(f"Failed to load dataset for model selection: {e}")
        st.stop()

    st.subheader("Model Selection")
    model_selection = st.selectbox(
        "Select a machine learning model:",
        ["Choose Model", "Random Forest", "Neural Networks"]
    )

    if model_selection == "Random Forest":
        st.subheader("Disaster Prediction with Random Forest Model")
        disaster_prediction_selection(data)

    elif model_selection == "Neural Networks":
        st.subheader("Neural Networks Model")
        neural_networks_selection(data)  # Call the function from services/neural_networks_selection

    elif model_selection == "Choose Model":
        st.write("Please select a model.")
