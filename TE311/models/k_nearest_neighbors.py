from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import numpy as np

def train_knn_model(data, target, k, disaster_type):
    # Predefined features based on the dataset
    predefined_features = [
        'Total Deaths', 'Latitude', 'Longitude', 'No. Injured',
        'Total Damage (\'000 US$)', 'Total Affected', 'Magnitude'
    ]

    # Ensure predefined features exist in the dataset
    for feature in predefined_features:
        if feature not in data.columns:
            raise ValueError(f"Feature '{feature}' is missing from the dataset.")

    # Filter data for the specific disaster type
    data_filtered = data[data[target] == disaster_type]

    if data_filtered.empty:
        raise ValueError(f"No data available for the selected disaster type: {disaster_type}")

    # Check if there are enough rows for training and testing
    if len(data_filtered) < 5:
        raise ValueError(f"Not enough data for disaster type '{disaster_type}' to train a KNN model.")

    # Split the data into features (X) and target (y)
    X = data_filtered[predefined_features]
    y = data_filtered[target]

    # Handle missing values by imputing them with the mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate a classification report
    class_report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "model": knn,
        "accuracy": accuracy,
        "y_test": y_test,
        "y_pred": y_pred,
        "classification_report": class_report
    }
