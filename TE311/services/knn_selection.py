import streamlit as st
from PIL import Image
import numpy as np
from dataset import load_data  # Ensure you implement `load_data` for your dataset
from models.k_nearest_neighbors import train_knn_model, save_knn_model, load_data


def predict_disaster(image, model, scaler, label_encoder=None, resize_dim=(32, 32)):
    # Preprocess the image (resize, flatten, scale)
    image = image.resize(resize_dim).convert('L')  # Resize and convert to grayscale
    features = np.array(image).flatten().reshape(1, -1)  # Flatten and reshape
    features = scaler.transform(features)  # Scale features

    # Predict disaster type
    prediction = model.predict(features)
    if label_encoder:
        prediction = label_encoder.inverse_transform(prediction)
    return prediction[0]


def knn_selection():
    st.title("Disaster Detection from Satellite Imagery")

    # Upload an image
    uploaded_image = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Option to train a new model
        if st.checkbox("Train a new KNN model"):
            with st.spinner("Training KNN model..."):
                features, labels = load_data(file_path=r'D:\csv file\disaster_sea.csv')  # Load dataset
                results = train_knn_model(features, labels)
                save_knn_model(results['model'], results['scaler'])
                st.success("New KNN model trained and saved successfully.")

        # Load pre-trained model
        try:
            model_data = load_data('D:\csv file\disaster_sea.csv')
            model = model_data['model']
            scaler = model_data['scaler']
            label_encoder = model_data.get('label_encoder')

            # Make prediction
            prediction = predict_disaster(image, model, scaler, label_encoder)
            st.success(f"Predicted Disaster Type: {prediction}")
        except (FileNotFoundError, EOFError) as e:
            st.error(str(e))
