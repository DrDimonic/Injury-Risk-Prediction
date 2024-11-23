import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    # Load data from a CSV file.
    return pd.read_csv(filepath)

# Process data for training
def preprocess_data(data):
      # Remove "Unnamed: 0" if it exists
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    features = data.drop(columns=['currently_injured']) 
    target = data['currently_injured']

    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    return normalized_features, target, scaler
