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
        model, predictions, test_loss = train_lstm(
            X_train, y_train, X_test, y_test,
            seq_length, input_size, hidden_size,
            num_layers, output_size, num_epochs, batch_size
        )
        
        st.success(f"Test Loss: {test_loss:.4f}")

        # Convert predictions into binary outcomes
        threshold = st.slider("Prediction Threshold", min_value=0.1, max_value=0.9, value=0.5)
        binary_predictions = ["Disaster Likely" if pred >= threshold else "Disaster Unlikely" for pred in predictions]

        # Display Predictions and Binary Results
        st.subheader("Prediction Results:")
        st.write(f"Predictions for {seq_length} years")
        results_df = pd.DataFrame({
            "Year": range(1, len(predictions) + 1),
            "Predicted Value": [pred.item() for pred in predictions],
            "Prediction": binary_predictions,
            "Actual Value": y_test.flatten()
        })
        st.write(results_df.head(20)) # Display the first 20 predictions

        # Explanation Section
        st.subheader("Explanation")
        if test_loss < 0.1:
            st.success(
                "The model achieved excellent performance with a very low test loss. "
                "Predictions are highly reliable, and the model effectively captured disaster patterns over time."
            )
        elif test_loss < 0.5:
            st.info(
                "The model shows moderate performance. Predictions align reasonably well with actual data, "
                "but some refinements may improve reliability."
            )
        else:
            st.warning(
                "The model's performance is suboptimal, with a high test loss. Predictions may not be very reliable. "
                "Consider revisiting data preprocessing, feature selection, or hyperparameter tuning."
            )
