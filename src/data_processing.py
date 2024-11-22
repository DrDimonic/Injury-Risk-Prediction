import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    # Load data from a CSV file.
    return pd.read_csv(filepath)

def preprocess_data(data):
    # Feature and target separation
    features = data.drop(columns=["currently_injured"])
    target = data["currently_injured"]

    # Apply SMOTE for balancing
    smote = SMOTE(random_state=42)
    features_balanced, target_balanced = smote.fit_resample(features, target)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_balanced, target_balanced, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test
