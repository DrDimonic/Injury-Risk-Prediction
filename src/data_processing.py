import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    # Load data from a CSV file.
    return pd.read_csv(filepath)

def preprocess_data(data):
    # Define the target column and feature set
    target_column = "currently_injured"
    features = data.drop(columns=[target_column])
    target = data[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    print(f"Initial training data distribution:\n{y_train.value_counts()}")

    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"Balanced training data distribution:\n{y_train_balanced.value_counts()}")

    return X_train_balanced, X_test, y_train_balanced, y_test

