from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def train_ensemble_model(data, prediction_input):
    """
    Predicts the type of disaster for a given country and date.

    Args:
    - data (pd.DataFrame): Input dataset.
    - prediction_input (dict): Contains 'Country', 'Start Year', 'Start Month', 'Start Day'.

    Returns:
    - dict: Predicted disaster type, probability, and model accuracy.
    """
    # Preprocessing
    data = data.fillna(0)

    # Filter data for the selected country
    country_data = data[data['Country'] == prediction_input["Country"]]

    # Features and Target
    features = ['Start Year', 'Start Month', 'Start Day', 'No. Affected',
                'Total Deaths', 'No. Injured', 'Total Damage (\'000 US$)', 'Total Affected']
    X = country_data[features]
    y = country_data['Disaster Type']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y_encoded)

    # Evaluate Model Accuracy
    y_pred = rf_model.predict(X_scaled)
    accuracy = accuracy_score(y_encoded, y_pred)

    # Prepare input for prediction
    input_data = pd.DataFrame([[
        prediction_input['Start Year'],
        prediction_input['Start Month'],
        prediction_input['Start Day'],
        0, 0, 0, 0, 0  # Default zeros for missing features
    ]], columns=features)

    input_scaled = scaler.transform(input_data)

    # Predict Disaster Type
    probabilities = rf_model.predict_proba(input_scaled)[0]
    predicted_class = rf_model.predict(input_scaled)[0]
    predicted_disaster = label_encoder.inverse_transform([predicted_class])[0]

    return {
        "predicted_disaster": predicted_disaster,
        "prediction_probability": max(probabilities),
        "model_accuracy": accuracy
    }
