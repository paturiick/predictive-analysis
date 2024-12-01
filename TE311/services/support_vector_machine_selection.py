import streamlit as st
import pandas as pd
from models.support_vector_machine import preprocess_data, train_svm, evaluate_model, save_model
from sklearn.metrics import classification_report

# Streamlit UI
def support_vector_machine_selection(data):
    if st.button("Train Support Vector Machine Model"):
        try:
            X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess_data(data)
        except ValueError as e:
            st.error(str(e))
            st.stop()
    
        svm_model = train_svm(X_train, y_train, C=1.0, gamma=0.1)
        # Evaluate the model
        accuracy, classication_report, confusion_matrix = evaluate_model(svm_model, X_test, y_test)
    
        st.success(f"Model Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        classification_dict = classification_report(y_test, svm_model.predict(X_test), output_dict=True)
        classification_df = pd.DataFrame(classification_dict,).transpose()
        st.write(classification_df)
        st.text("Confusion Matrix:")
        st.write(confusion_matrix)
    
    # Save the model
        save_option = st.checkbox("Save the trained model?")
        if save_option:
            save_model(svm_model, label_encoder, scaler)
            st.success("Model and metadata saved successfully!")
