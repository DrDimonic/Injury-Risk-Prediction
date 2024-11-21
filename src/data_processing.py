import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Preprocess the data: handle missing values, scale features, etc."""
    data = data.dropna()  # Drop rows with missing values
    features = data.drop(columns=['injury_risk'])
    target = data['injury_risk']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return train_test_split(features_scaled, target, test_size=0.2, random_state=42)

