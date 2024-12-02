from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import pickle


def train_knn_model(features, labels, k=5, metric='euclidean', weights='uniform'):
    # Encode labels if they are categorical
    if labels.dtype == 'object':
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
    else:
        label_encoder = None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the KNN classifier
    knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    knn_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return {
        "model": knn_model,
        "scaler": scaler,
        "label_encoder": label_encoder,  # Save encoder if used
        "accuracy": accuracy,
        "classification_report": pd.DataFrame(class_report).transpose(),
        "confusion_matrix": conf_matrix
    }


def save_knn_model(model, scaler, label_encoder=None, file_path='D:\csv file\disaster_sea.csv'):
    model_data = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder
    }
    with open(file_path, 'wb') as file:
        pickle.dump(model_data, file)
    print(f"Model saved successfully to {file_path}")


def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.fillna(0)
    # Use 'Disaster Type' as labels and other relevant columns as features
    features = data[['Total Deaths', 'No. Injured', 'No. Affected', 'No. Homeless', 'Total Damage (\'000 US$)']].fillna(0)
    labels = data['Disaster Type']
    
    return features, labels