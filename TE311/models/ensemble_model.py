from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def train_ensemble_model(data, prediction_input, model_type='random_forest'):
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

    # Train the chosen model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)  # Enable probability predictions
    else:
        raise ValueError("Invalid model type. Choose 'random_forest' or 'svm'.")

    model.fit(X_scaled, y_encoded)

    # Model Evaluation Metrics
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y_encoded, y_pred)
    precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)

    # Prepare input for prediction
    input_data = pd.DataFrame([[ 
        prediction_input['Start Year'],
        prediction_input['Start Month'],
        prediction_input['Start Day'],
        0, 0, 0, 0, 0  # Default zeros for missing features
    ]], columns=features)

    input_scaled = scaler.transform(input_data)

    # Predict Disaster Type
    probabilities = model.predict_proba(input_scaled)[0]
    predicted_class = model.predict(input_scaled)[0]
    predicted_disaster = label_encoder.inverse_transform([predicted_class])[0]

    # Return prediction and evaluation metrics
    return {
        "predicted_disaster": predicted_disaster,
        "prediction_probability": max(probabilities),
        "model_accuracy": accuracy,
        "model_precision": precision,
        "model_recall": recall,
        "model_f1_score": f1
    }
