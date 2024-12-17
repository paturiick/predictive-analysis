from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def train_ensemble_model(data, prediction_input):
    # Preprocessing
    data = data.fillna(0)

    # Feature Engineering
    data['Event Duration'] = (
        (data['End Year'] - data['Start Year']) * 365 +
        (data['End Month'] - data['Start Month']) * 30 +
        (data['End Day'] - data['Start Day'])
    )
    data['Event Duration'] = data['Event Duration'].clip(lower=0)  # Ensure no negative durations
    data['Affected to Deaths Ratio'] = data['No. Affected'] / (data['Total Deaths'] + 1)
    data['Damage per Affected'] = data['Total Damage (\'000 US$)'] / (data['No. Affected'] + 1)
    data['Severe Impact Flag'] = (data['No. Affected'] > 1e6).astype(int)  # Binary flag for major disasters

    # Filter data for the selected country
    country_data = data[data['Country'] == prediction_input["Country"]]

    # Features and Target
    features = [
        'Start Year', 'Start Month', 'Start Day', 'No. Affected',
        'Total Deaths', 'No. Injured', 'Total Damage (\'000 US$)', 'Total Affected',
        'Event Duration', 'Affected to Deaths Ratio', 'Damage per Affected', 'Severe Impact Flag'
    ]
    X = country_data[features]
    y = country_data['Disaster Type']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
        0, 0, 0, 0, 0,
        0, 0, 0, 0
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
