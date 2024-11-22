import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

def load_data(filepath):
    # Load data from a CSV file.
    return pd.read_csv(filepath)

def preprocess_data(data):
    # Define the target column and feature set
    data = data.dropna()  # Handle missing values
    target = data['currently_injured']
    features = data.drop(columns=['currently_injured'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_balanced, y_test

