import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def preprocess_data(data):
    
    # Fill missing values
    data = data.fillna(0)
    
    # Identify non-numeric columns
    non_numeric_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Encode non-numeric columns
    for column in non_numeric_columns:
        data[column] = data[column].astype('category').cat.codes
    
    # Ensure target column ('Disaster Type') is encoded
    if 'Disaster Type' in data.columns:
        label_encoder = LabelEncoder()
        data['Disaster Type'] = label_encoder.fit_transform(data['Disaster Type'])
    else:
        raise ValueError("Dataset must contain a 'Disaster Type' column.")
    
    # Separate features and target
    X = data.drop('Disaster Type', axis=1)
    y = data['Disaster Type']
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder, scaler

def train_svm(X_train, y_train, C=1.0, gamma=0.1):
    # Initialize the SVM model
    svm_model = SVC(kernel='rbf', C=C, gamma=gamma)
    
    # Train the model
    svm_model.fit(X_train, y_train)
    
    return svm_model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and generate reports
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm

def save_model(model, label_encoder, scaler, model_path="svm_model.pkl", metadata_path="svm_metadata.pkl"):
    # Save the trained model, label encoder, and scaler
    joblib.dump(model, model_path)
    joblib.dump({"label_encoder": label_encoder, "scaler": scaler}, metadata_path)

def load_model(model_path="svm_model.pkl", metadata_path="svm_metadata.pkl"):
    # Load the trained model, label encoder, and scaler
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata["label_encoder"], metadata["scaler"]
