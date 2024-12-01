import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = r'D:\csv file\disaster_sea.csv'
df = pd.read_csv(file_path)

# Drop missing target values
df = df.dropna(subset=['Disaster Type'])

# Impute missing numerical values
numerical_columns = ['Total Damage (\'000 US$)', 'Total Affected', 'Start Year', 'Start Month']
imputer = SimpleImputer(strategy='mean')
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

# Handle missing categorical values (if any)
categorical_columns = ['Country']
df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# Encode categorical variables
label_encoder = LabelEncoder()
df['Disaster Type'] = label_encoder.fit_transform(df['Disaster Type'])
df['Country'] = label_encoder.fit_transform(df['Country'])

# Normalize numerical features
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Select features and target
features = ['Start Year', 'Start Month', 'Country'] + numerical_columns
X = df[features]
y = df['Disaster Type']

# Verify no NaN values remain in X
if X.isnull().values.any():
    raise ValueError("NaN values found in features after preprocessing")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
k = 5  # Choose the number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", report)
