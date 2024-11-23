import pandas as pd
from sklearn.preprocessing import normalize

def load_data(filepath):
    # Load data from a CSV file.
    return pd.read_csv(filepath)

# Process data for training
def preprocess_data(data):
    # Seperate features and target
    features = data.drop(columns=['currently_injured']) 
    target = data['currently_injured']

    # Normalize features (L1)
    normalized_features = normalize(features, norm='l1', axis=0)

    return normalized_features, target
