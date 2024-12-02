import streamlit as st
import pandas as pd
from models.neural_networks import preprocess_and_create_sequences, train_lstm
from sklearn.model_selection import train_test_split

def neural_networks_selection(data):
    
    st.title("Neural Networks for Time-Series Prediction")
    
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

        # Ensure consistent lengths for all arrays
        display_length = min(seq_length, len(predictions), len(y_test))
        predictions = predictions[:display_length]
        y_test = y_test[:display_length]
        years = range(1, display_length + 1)

        # Convert predictions into a DataFrame
        results_df = pd.DataFrame({
            "Year": [f"Year {i}" for i in years],
            "Predicted Value": [pred.item() for pred in predictions],
            "Actual Value": y_test.flatten()
        })
        st.write(results_df)

        # Visualize Predictions
        st.line_chart(results_df.set_index("Year"))

        # Explanation Section
        st.subheader("Results")
        differences = abs(results_df["Predicted Value"] - results_df["Actual Value"])
        avg_difference = differences.mean()
        
        if avg_difference < 0.1:
            st.success(
                f"The predictions are highly accurate with an average difference of {avg_difference:.2f}. "
                "This indicates that the model effectively learned the patterns and relationships in the data.",
            )
        elif avg_difference < 0.5:
            st.info(
                f"The predictions are moderately accurate with an average difference of {avg_difference:.2f}. "
                "The model captures trends reasonably well but may require tuning for improved precision.",
            )
        else:
            st.warning(
                f"The predictions have a high average difference of {avg_difference:.2f}, "
                "indicating that the model struggles to align closely with the actual values. ",
            )