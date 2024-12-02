from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def classification_report_as_dataframe(report):
    report_dict = classification_report(
        report['y_true'], 
        report['y_pred'], 
        target_names=["No", "Yes"], 
        output_dict=True, 
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df.reset_index()

def train_ensemble_model(data, disaster_type):
    # Preprocessing
    data = data.fillna(0)

    # Encode the disaster type
    label_encoder = LabelEncoder()
    data['Disaster Type Encoded'] = label_encoder.fit_transform(data['Disaster Type'])
    disaster_encoded = label_encoder.transform([disaster_type])[0]

    # Add a column to indicate disaster occurrence
    data['Disaster Occurrence'] = (data['Disaster Type Encoded'] == disaster_encoded).astype(int)

    # Group by country and calculate disaster frequency
    country_disaster_stats = (
        data.groupby('Country')['Disaster Occurrence']
        .agg(['sum', 'count'])
        .rename(columns={'sum': 'Disaster Count', 'count': 'Total Records'})
        .reset_index()
    )
    country_disaster_stats['Disaster Frequency'] = country_disaster_stats['Disaster Count'] / country_disaster_stats['Total Records']

    # Merge the frequency stats back into the main data
    data = pd.merge(data, country_disaster_stats[['Country', 'Disaster Frequency']], on='Country', how='left')

    # Features and target
    features = ['Start Year', 'Start Month', 'Start Day', 'No. Affected',
                'Total Deaths', 'No. Injured', 'Total Damage (\'000 US$)', 
                'Total Affected', 'Disaster Frequency']
    X = data[features]
    y = data['Disaster Occurrence']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check if the disaster type has more than one class
    if y.nunique() <= 1:
        raise ValueError(
            f"The selected disaster type '{disaster_type}' has only one class and cannot be used for training."
        )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Train SVM
    svm_model = SVC(probability=True, kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)

    # Ensemble Predictions (Soft Voting)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    svm_probs = svm_model.predict_proba(X_test)[:, 1]
    ensemble_probs = (rf_probs + svm_probs) / 2

    # Final Predictions
    y_pred = (ensemble_probs > 0.5).astype(int)

    # Evaluate Ensemble Model
    accuracy = accuracy_score(y_test, y_pred)
    class_report = {
        "y_true": y_test,
        "y_pred": y_pred
    }

    # Feature Importances
    feature_importances = dict(zip(features, rf_model.feature_importances_))

    # Country-Specific Predictions
    country_data = data[['Country'] + features].drop_duplicates()
    country_data_scaled = scaler.transform(country_data[features])
    rf_country_probs = rf_model.predict_proba(country_data_scaled)[:, 1]
    svm_country_probs = svm_model.predict_proba(country_data_scaled)[:, 1]
    country_data['Prediction Probability'] = (rf_country_probs + svm_country_probs) / 2
    country_data['Prediction'] = country_data['Prediction Probability'].apply(
        lambda prob: disaster_type if prob > 0.5 else f"No {disaster_type}"
    )

    return {
        "accuracy": accuracy,
        "classification_report": classification_report_as_dataframe(class_report),
        "feature_importances": feature_importances,
        "country_predictions": country_data[['Country', 'Prediction', 'Prediction Probability']].sort_values(by='Prediction Probability', ascending=False)
    }
