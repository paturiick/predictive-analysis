import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def classification_report_as_dataframe(report):
    report_dict = classification_report(
        report['y_true'], 
        report['y_pred'], 
        target_names=["No", "Yes"], 
        output_dict=True, 
        zero_division=0
    )
    
    # Transform into a DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df.reset_index()


def train_random_forest_model(data, disaster_type):
    # Preprocessing
    data = data.fillna(0)

    # Encode the target variable and filter by the specified disaster type
    label_encoder = LabelEncoder()
    data['Disaster Type Encoded'] = label_encoder.fit_transform(data['Disaster Type'])
    disaster_encoded = label_encoder.transform([disaster_type])[0]

    # Create a binary target: whether the disaster occurs or not
    data['Disaster Occurrence'] = (data['Disaster Type Encoded'] == disaster_encoded).astype(int)

    # Check if the disaster type has more than one class
    if data['Disaster Occurrence'].nunique() <= 1:
        raise ValueError(
            f"The selected disaster type '{disaster_type}' has only one class and cannot be used for training."
        )

    # Features and target
    features = ['Total Deaths', 'No. Injured', 'Total Damage (\'000 US$)', 'Total Affected']
    X = data[features]
    y = data['Disaster Occurrence']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    class_report = {
        "y_true": y_test,
        "y_pred": y_pred
    }
    feature_importances = dict(zip(features, model.feature_importances_))

    return {
        "accuracy": accuracy,
        "classification_report": classification_report_as_dataframe(class_report),
        "feature_importances": feature_importances,
    }