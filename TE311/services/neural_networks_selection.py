import streamlit as st
import pandas as pd
from models.neural_networks import preprocess_and_create_sequences, train_lstm
from sklearn.model_selection import train_test_split


def neural_networks_selection(data):
    seq_length = st.slider("Years to Predict", min_value=2, max_value=10, value=5)
    hidden_size = st.slider("Hidden Size", min_value=32, max_value=256, value=128)
    num_layers = st.slider("Number of Layers", min_value=1, max_value=5, value=3)
    num_epochs = st.slider("Number of Epochs", min_value=10, max_value=100, value=50)
    batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=32)

    # Preprocess the data and create sequences
    X, y, scaler = preprocess_and_create_sequences(data, seq_length)
    input_size = X.shape[2]
    output_size = 1

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    if st.button("Train and Evaluate Model"):
        model, predictions, test_loss = train_lstm(X_train, y_train, X_test, y_test, seq_length, input_size, hidden_size, num_layers, output_size, num_epochs, batch_size)
        st.success(f"Test Loss: {test_loss:.4f}")

        # Show Predictions
        st.subheader("Predictions")
        st.write("Sample Predictions (First 10):")
        for i in range(min(10, len(predictions))):
            st.write(f"Predicted: {predictions[i].item():.2f}, Actual: {y_test[i].item():.2f}")

         # Explanation Section
        st.subheader("Explanation")
        if test_loss < 0.1:
            st.success(
                "The model achieved excellent performance with a very low test loss. "
                "Predictions closely align with the actual values, indicating that the LSTM effectively captured temporal patterns."
            )
        elif test_loss < 0.5:
            st.info(
                "The model shows moderate performance. While some predictions align with actual values, "
                "others deviate. This suggests the model partially captured the temporal dependencies, "
                "but further tuning or additional data may improve results."
            )
        else:
            st.warning(
                "The model performance is suboptimal. High test loss indicates difficulty in capturing the patterns in the data. "
                "Consider refining features, adding more data, or optimizing hyperparameters."
            )