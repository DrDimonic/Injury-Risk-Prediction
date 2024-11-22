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

    # Apply SMOTE-ENN
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(features, target)

    return X_resampled, y_resampled

