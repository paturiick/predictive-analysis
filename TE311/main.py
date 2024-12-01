import streamlit as st
import os
import pandas as pd

config_content = """
[theme]
primaryColor="#948979"
backgroundColor="#153448"
secondaryBackgroundColor="#3c5b6f"
textColor="#dfd0b8"
"""

st.set_page_config(layout="wide")
os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/config.toml", "w") as config_file:
    config_file.write(config_content)
    
st.title("Predictive Analysis for Natural Disaster Management")
# Preload the dataset
file_path = r'D:\csv file\disaster_sea.csv'

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
