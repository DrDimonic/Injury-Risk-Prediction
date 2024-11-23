import pandas as pd
from sklearn.preprocessing import normalize


def load_data(filepath):
    # Load data from a CSV file.
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Preprocess data for training."""
    features = data.drop(columns=['currently_injured'])  # Drop target column
    target = data['currently_injured']

    # Apply L1 normalization
    features_normalized = normalize(features, norm='l1', axis=1)

    return features_normalized, target
