import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    # Define target variable
    target = data['currently_injured']
    
    # Drop the target and other non-feature columns
    features = data.drop(columns=['currently_injured', 'n_injuries', 'n_severe_injuries'])
    
    # Return features and target
    return features, target

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return train_test_split(features_scaled, target, test_size=0.2, random_state=42)

